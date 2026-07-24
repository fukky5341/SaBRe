## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 86.514199010344


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-31.8210735, 70.5720062, -31.8210735, 70.5720062, -102.3930817, 102.3930817)
1: (-66.6505203, 105.5066299, -66.6505203, 105.5066299, -172.1571503, 172.1571503)
2: (-50.6369743, 103.4140015, -50.6369743, 103.4140015, -154.0509644, 154.0509644)
3: (-76.9388275, 123.2527008, -76.9388275, 123.2527008, -200.1915283, 200.1915283)
4: (-70.2549973, 117.4171600, -70.2549973, 117.4171600, -187.6721497, 187.6721497)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.84 + 2.55 = 3.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -86.5211207, upper bound: 86.5211207

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5197899, upper bound: 86.5190158
time: 0.99 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5189021, upper bound: 86.5189021
time: 0.65 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.72 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 0, lower bound: -86.5197899, upper bound: 86.5190158
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 0, lower bound: -86.5189021, upper bound: 86.5189021

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -28.6602478, 63.5184097, -30.4394760, 67.4860992, -96.1463470, 93.9578629
1: -59.4984627, 94.9266205, -63.5261230, 100.8758240, -160.3742676, 158.4527130
2: -45.4671211, 92.3517075, -48.3791466, 98.5660019, -144.0331116, 140.7308502
3: -69.0993729, 110.2785263, -73.5126038, 117.5585938, -186.6579590, 183.7911377
4: -63.3898087, 105.2295685, -67.2546158, 112.0820389, -175.4718475, 172.4841156

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5189021, upper bound: 86.5189021
time: 0.87 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5189021, upper bound: 86.5189021
time: 0.67 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -37.7722359, 83.6511536, -31.0233154, 68.8542938, -106.6265259, 114.6744690
1: -77.5871658, 124.5004730, -64.9477005, 102.9474869, -180.5346527, 189.4481812
2: -59.5412712, 120.8561783, -49.3584099, 100.9164200, -160.4576874, 170.2145844
3: -90.5640182, 144.2913818, -74.9898911, 120.2925568, -210.8565674, 219.2812500
4: -83.2820587, 137.9900970, -68.5070801, 114.5673141, -197.8493652, 206.4971466

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5173999, upper bound: 86.5183371
time: 1.15 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169788, upper bound: 86.5169788
time: 0.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.79 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 0, lower bound: -86.5189021, upper bound: 86.5189021
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 0, lower bound: -86.5189021, upper bound: 86.5189021
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 0, lower bound: -86.5173999, upper bound: 86.5183371
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 0, lower bound: -86.5169788, upper bound: 86.5169788

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -28.6602478, 63.5184097, -28.6602478, 63.5184097, -92.1786499, 92.1786575
1: -59.4984627, 94.9266205, -59.4984627, 94.9266205, -154.4250793, 154.4250793
2: -45.4671211, 92.3517075, -45.4671211, 92.3517075, -137.8188019, 137.8188019
3: -69.0993729, 110.2785263, -69.0993729, 110.2785263, -179.3778992, 179.3778992
4: -63.3898087, 105.2295685, -63.3898087, 105.2295685, -168.6193085, 168.6193085

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5186990, upper bound: 86.5184206
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5187986, upper bound: 86.5180495
time: 0.67 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -28.6602478, 63.5184097, -37.7722359, 83.6511536, -112.3114014, 101.2906342
1: -59.4984627, 94.9266205, -77.5871658, 124.5004730, -183.9989319, 172.5137939
2: -45.4671211, 92.3517075, -59.5412712, 120.8561783, -166.3233032, 151.8929596
3: -69.0993729, 110.2785263, -90.5640182, 144.2913818, -213.3907471, 200.8425446
4: -63.3898087, 105.2295685, -83.2820587, 137.9900970, -201.3798523, 188.5115967

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5186990, upper bound: 86.5184206
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5187986, upper bound: 86.5180495
time: 0.67 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -37.4973030, 83.0148239, -28.2654629, 62.4260864, -99.9233856, 111.2802887
1: -76.9270325, 123.5443497, -58.4345589, 93.2649155, -170.1919403, 181.9789124
2: -59.0771751, 119.8678665, -44.7808723, 90.6591492, -149.7363281, 164.6487427
3: -89.8692780, 143.1183472, -68.0360336, 108.1961823, -198.0654144, 211.1543579
4: -82.6814804, 136.8724213, -62.5292778, 103.3136292, -185.9951172, 199.4017029

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169788, upper bound: 86.5169788
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169788, upper bound: 86.5169788
time: 1.04 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -36.8594131, 81.6456757, -42.2469940, 93.3343430, -129.9284210, 123.4890366
1: -75.5382614, 121.5027084, -86.0541077, 139.2621307, -214.8003845, 207.5568237
2: -58.0658531, 117.8076096, -66.3438797, 134.4250031, -192.4908447, 184.1514893
3: -88.2619171, 140.7279968, -100.8490372, 160.9217377, -249.1836548, 241.5770264
4: -81.2918091, 134.5743713, -93.0445023, 153.9315338, -234.9144592, 226.9726257

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169788, upper bound: 86.5169788
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169788, upper bound: 86.5169788
time: 1.03 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.81 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 0, lower bound: -86.5186990, upper bound: 86.5184206
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 0, lower bound: -86.5187986, upper bound: 86.5180495
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 0, lower bound: -86.5186990, upper bound: 86.5184206
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 0, lower bound: -86.5187986, upper bound: 86.5180495
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 0, lower bound: -86.5169788, upper bound: 86.5169788
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 0, lower bound: -86.5169788, upper bound: 86.5169788
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 0, lower bound: -86.5169788, upper bound: 86.5169788
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 0, lower bound: -86.5169788, upper bound: 86.5169788

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -27.6749191, 61.3052101, -28.6127548, 63.4120407, -91.0869446, 89.9179688
1: -57.3225365, 91.5926590, -59.3936615, 94.7663193, -152.0888519, 150.9863129
2: -43.8713226, 89.0364456, -45.3903656, 92.1922836, -136.0635986, 134.4268036
3: -66.6552582, 106.3389359, -68.9814758, 110.0887909, -176.7440338, 175.3204041
4: -61.2314949, 101.4781342, -63.2858887, 105.0492325, -166.2807312, 164.7640228

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169792, upper bound: 86.5165862
time: 1.29 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151250, upper bound: 86.5154501
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -27.8543739, 61.7165260, -28.4315224, 62.9854050, -90.8397827, 90.1480484
1: -57.6641655, 92.2571945, -58.9777870, 94.1255035, -151.7896729, 151.2349854
2: -44.1420746, 89.6542816, -45.0921249, 91.5561905, -135.6982727, 134.7463989
3: -67.0748291, 107.1058578, -68.5252228, 109.3343811, -176.4092102, 175.6310730
4: -61.6210403, 102.1806717, -62.8873520, 104.3219452, -165.9429626, 165.0680084

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169899, upper bound: 86.5162833
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151077, upper bound: 86.5151077
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -27.6749191, 61.3052101, -37.7254372, 83.5455399, -111.2204590, 99.0306473
1: -57.3225365, 91.5926590, -77.4832153, 124.3404236, -181.6629639, 169.0758667
2: -43.8713226, 89.0364456, -59.4650612, 120.6983871, -164.5697021, 148.5014954
3: -66.6552582, 106.3389359, -90.4477081, 144.1026459, -210.7578735, 196.7866516
4: -61.2314949, 101.4781342, -83.1791229, 137.8100891, -199.0415802, 184.6572571

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5186990, upper bound: 86.5179070
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5186990, upper bound: 86.5180495
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -27.8543739, 61.7165260, -37.5434456, 83.1196747, -110.9740448, 99.2599716
1: -57.6641655, 92.2571945, -77.0659943, 123.7063370, -181.3704987, 169.3231812
2: -44.1420746, 89.6542816, -59.1643906, 120.0704346, -164.2124939, 148.8186646
3: -67.0748291, 107.1058578, -89.9897156, 143.3571625, -210.4319916, 197.0955658
4: -61.6210403, 102.1806717, -82.7766647, 137.0890350, -198.7100677, 184.9573212

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5182946, upper bound: 86.5173257
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5160518, upper bound: 86.5160532
time: 1.32 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -35.3220711, 78.0994186, -28.2654629, 62.4260864, -97.7481537, 106.3648834
1: -71.8422546, 116.2362137, -58.4345589, 93.2649155, -165.1071625, 174.6707764
2: -55.4358864, 112.2269745, -44.7808723, 90.6591492, -146.0950012, 157.0078430
3: -84.4456253, 134.1693726, -68.0360336, 108.1961823, -192.6417542, 202.2053833
4: -77.9337540, 128.3165588, -62.5292778, 103.3136292, -181.2473755, 190.8457947

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5173999, upper bound: 86.5183371
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5173999, upper bound: 86.5183371
time: 1.24 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -48.7698364, 107.6354675, -28.2654629, 62.4260864, -110.6844940, 135.4063721
1: -98.0828018, 160.2976227, -58.4345589, 93.2649155, -191.3477173, 218.4312134
2: -76.0840988, 153.9630737, -44.7808723, 90.6591492, -166.4676208, 198.7439423
3: -115.8425217, 184.6167603, -68.0360336, 108.1961823, -223.4495239, 252.6527710
4: -107.3829651, 176.5917816, -62.5292778, 103.3136292, -209.3907471, 238.7311096

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5173999, upper bound: 86.5183371
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5173999, upper bound: 86.5183371
time: 1.15 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -35.3220711, 78.0994186, -42.1068077, 92.9918442, -128.0109863, 119.7537842
1: -71.8422546, 116.2362137, -85.7729340, 138.7287292, -210.5709686, 202.0091553
2: -55.4358864, 112.2269745, -66.1236649, 133.9476318, -189.3835144, 178.3506470
3: -84.4456253, 134.1693726, -100.5109024, 160.3229218, -244.7685242, 234.6802673
4: -77.9337540, 128.3165588, -92.7386703, 153.3672638, -230.8990173, 220.2977142

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5155277, upper bound: 86.5162969
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5152294, upper bound: 86.5152294
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -49.3108788, 108.9386978, -42.2469940, 93.3343430, -141.6334076, 150.0471191
1: -99.2558975, 162.3510895, -86.0541077, 139.2621307, -238.2005463, 247.7566833
2: -76.9432755, 155.9279480, -66.3438797, 134.4250031, -210.9823608, 221.9160767
3: -117.1908417, 187.0321503, -100.8490372, 160.9217377, -277.3742981, 287.3575439
4: -108.5717087, 178.8141174, -93.0445023, 153.9315338, -260.4354248, 269.9377441

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5155277, upper bound: 86.5162969
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5152294, upper bound: 86.5152294
time: 0.66 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.46 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -86.5169792, upper bound: 86.5165862
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -86.5151250, upper bound: 86.5154501
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -86.5169899, upper bound: 86.5162833
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -86.5151077, upper bound: 86.5151077
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -86.5186990, upper bound: 86.5179070
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -86.5186990, upper bound: 86.5180495
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -86.5182946, upper bound: 86.5173257
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -86.5160518, upper bound: 86.5160532
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -86.5173999, upper bound: 86.5183371
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -86.5173999, upper bound: 86.5183371
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -86.5173999, upper bound: 86.5183371
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -86.5173999, upper bound: 86.5183371
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -86.5155277, upper bound: 86.5162969
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -86.5152294, upper bound: 86.5152294
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -86.5155277, upper bound: 86.5162969
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -86.5152294, upper bound: 86.5152294

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -27.5745831, 61.0754089, -27.9589462, 61.9158974, -89.4904785, 89.0343552
1: -57.1126251, 91.2481003, -58.0258255, 92.5207672, -149.6333771, 149.2739258
2: -43.7119446, 88.7050171, -44.3505783, 90.0344009, -133.7463226, 133.0556030
3: -66.4137115, 105.9407272, -67.4060364, 107.4967422, -173.9104309, 173.3467407
4: -61.0110283, 101.0981827, -61.8487091, 102.5747604, -163.5857849, 162.9468689

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151250, upper bound: 86.5154500
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151250, upper bound: 86.5154501
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -27.1459045, 60.1593857, -32.4531136, 71.4387512, -98.5846558, 92.6124954
1: -56.2073517, 89.8687286, -68.0326691, 106.7323532, -162.9396820, 157.9013977
2: -43.0449371, 87.3070374, -51.7384949, 104.5068665, -147.5517883, 139.0455322
3: -65.4018326, 104.2749100, -78.5674896, 124.2988205, -189.7006531, 182.8424072
4: -60.0861435, 99.5445099, -71.6982498, 118.8128815, -178.8990173, 171.2427673

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5114215, upper bound: 86.5122361
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5110740, upper bound: 86.5121682
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -27.7527828, 61.4841766, -27.7809429, 61.4963074, -89.2490845, 89.2651215
1: -57.4521370, 91.9089737, -57.6178703, 91.8910065, -149.3431396, 149.5268250
2: -43.9808998, 89.3194351, -44.0578232, 89.4094009, -133.3903046, 133.3772583
3: -66.8303528, 106.7042084, -66.9583282, 106.7554932, -173.5858459, 173.6625366
4: -61.3977699, 101.7967834, -61.4574432, 101.8600998, -163.2578583, 163.2541962

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151077, upper bound: 86.5151077
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151077, upper bound: 86.5151077
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -27.3564491, 60.6395187, -32.2891273, 71.0513763, -98.4078064, 92.9286499
1: -56.6136475, 90.6367188, -67.6542282, 106.1540451, -162.7677002, 158.2909393
2: -43.3648453, 88.0272522, -51.4688416, 103.9361954, -147.3010406, 139.4960632
3: -65.8984070, 105.1632233, -78.1541138, 123.6150665, -189.5134125, 183.3173218
4: -60.5433884, 100.3634338, -71.3378372, 118.1601715, -178.7035370, 171.7012634

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5112280, upper bound: 86.5119988
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5087855, upper bound: 86.5087855
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -27.6749191, 61.3052101, -36.7977562, 81.4488297, -109.1237488, 98.1029663
1: -57.3225365, 91.5926590, -75.4232635, 121.1758881, -178.4984283, 167.0159149
2: -43.8713226, 89.0364456, -57.9546890, 117.5645905, -161.4359131, 146.9911346
3: -66.6552582, 106.3389359, -88.1447372, 140.3554535, -207.0106964, 194.4836426
4: -61.2314949, 101.4781342, -81.1409607, 134.2484741, -195.4799652, 182.6190948

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5158654, upper bound: 86.5165058
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5160456, upper bound: 86.5164353
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -27.6749191, 61.3052101, -36.9206200, 81.7408447, -109.4157639, 98.2258301
1: -57.3225365, 91.5926590, -75.6566391, 121.6775818, -179.0000916, 167.2492828
2: -43.8713226, 89.0364456, -58.1397781, 118.0272903, -161.8986206, 147.1762085
3: -66.6552582, 106.3389359, -88.4308701, 140.9516754, -207.6069183, 194.7697754
4: -61.2314949, 101.4781342, -81.4077530, 134.7594299, -195.9909210, 182.8858795

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5158654, upper bound: 86.5165058
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5160456, upper bound: 86.5164353
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -27.2341728, 60.4301910, -38.1483078, 84.8693771, -112.1035461, 98.5784836
1: -56.4045677, 90.3409348, -78.3411942, 126.4909897, -182.8955536, 168.6820984
2: -43.1701126, 87.7779770, -60.1318054, 122.5396271, -165.7097321, 147.9097900
3: -65.5899124, 104.8860016, -91.4803162, 146.4525452, -212.0424500, 196.3663177
4: -60.2609444, 100.0593185, -84.1040802, 140.0159149, -200.2768555, 184.1633911

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5177835, upper bound: 86.5167562
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5180195, upper bound: 86.5170243
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -27.6645031, 61.3086395, -36.9628029, 81.8988342, -109.5633240, 98.2714233
1: -57.2522354, 91.6427536, -75.8259735, 121.8751602, -179.1273651, 167.4687195
2: -43.8358383, 89.0458679, -58.2292061, 118.2659454, -162.1017761, 147.2750702
3: -66.6098862, 106.3831024, -88.5776749, 141.2120667, -207.8219299, 194.9607849
4: -61.2020454, 101.4936752, -81.4928894, 135.0363770, -196.2384033, 182.9865723

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5158643, upper bound: 86.5160295
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5158643, upper bound: 86.5160532
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -35.3220711, 78.0994186, -26.3504200, 58.1767349, -93.4988098, 104.4498367
1: -71.8422546, 116.2362137, -54.0193672, 86.9120712, -158.7543335, 170.2555847
2: -55.4358864, 112.2269745, -41.6006546, 83.9912643, -139.4271240, 153.8276215
3: -84.4456253, 134.1693726, -63.2661858, 100.3837051, -184.8292694, 197.4355469
4: -77.9337540, 128.3165588, -58.3637619, 95.9337921, -173.8675079, 186.6803284

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5176377, upper bound: 86.5181877
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5177355, upper bound: 86.5177355
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -35.3220711, 78.0994186, -35.3220367, 78.0993195, -113.1423264, 113.1423874
1: -71.8422546, 116.2362137, -71.8421631, 116.2360764, -188.0783081, 188.0783691
2: -55.4358864, 112.2269745, -55.4358253, 112.2268219, -167.6627045, 167.6627960
3: -84.4456253, 134.1693726, -84.4455414, 134.1691895, -218.6147919, 218.6148834
4: -77.9337540, 128.3165588, -77.9337006, 128.3163605, -205.9299316, 205.9300079

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5176377, upper bound: 86.5181877
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5177355, upper bound: 86.5177355
time: 1.23 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -48.6589890, 107.3824997, -26.3504200, 58.1767349, -106.3373489, 133.2576599
1: -97.8470001, 159.9006348, -54.0193672, 86.9120712, -184.7590637, 213.5943756
2: -75.9108963, 153.5730896, -41.6006546, 83.9912643, -159.5064087, 195.1737366
3: -115.5692444, 184.1391144, -63.2661858, 100.3837051, -215.2857666, 247.4053040
4: -107.1397247, 176.1630554, -58.3637619, 95.9337921, -201.7643738, 234.1551208

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169131, upper bound: 86.5169406
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157348, upper bound: 86.5166083
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -49.0459785, 108.2926559, -35.3220367, 78.0993195, -126.1757202, 142.6756287
1: -98.7047119, 161.3455963, -71.8421631, 116.2360764, -214.7066956, 232.4994507
2: -76.5252762, 154.9968719, -55.4358253, 112.2268219, -188.2589569, 210.3000641
3: -116.5435410, 185.8847198, -84.4455414, 134.1691895, -249.9231262, 270.0471191
4: -107.9903183, 177.7281647, -77.9337006, 128.3163605, -234.3586426, 254.2336884

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169131, upper bound: 86.5169406
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157348, upper bound: 86.5166083
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -34.8847122, 77.1361389, -40.9864769, 90.4881897, -125.0503616, 117.6442413
1: -70.8946304, 114.7803497, -83.3536606, 134.9154968, -205.8101196, 198.1340027
2: -54.7284088, 110.7854462, -64.3174057, 130.2234192, -184.9518280, 175.1028442
3: -83.3726196, 132.4660645, -97.7592773, 155.9017334, -239.2743530, 230.2253418
4: -76.9691162, 126.6858902, -90.2625046, 149.1202087, -225.6419983, 216.1130676

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145153, upper bound: 86.5162840
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5154599, upper bound: 86.5150349
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5154599, upper bound: 86.5150413
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -34.3422165, 76.0354385, -40.3365173, 89.0785980, -123.1259232, 115.9098587
1: -69.7409821, 113.1333618, -81.8373032, 132.8713684, -202.6123505, 194.9706116
2: -53.8469543, 109.1300507, -63.2025185, 128.0898743, -181.9368286, 172.3325653
3: -82.0537033, 130.5261383, -96.1190033, 153.4411011, -235.4947815, 226.6451263
4: -75.7649002, 124.8354263, -88.7852783, 146.7519684, -222.1168365, 212.8821106

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5154599, upper bound: 86.5149905
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5154599, upper bound: 86.5149788
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -48.8849449, 107.9920883, -41.1501846, 90.8865433, -138.7345886, 147.9737091
1: -98.3257065, 160.9142609, -83.6801300, 135.5353394, -233.5190887, 243.9036560
2: -76.2519684, 154.5153351, -64.5733566, 130.7780609, -206.5956268, 218.6860352
3: -116.1411667, 185.3589325, -98.1520233, 156.5973969, -271.9711914, 282.9386902
4: -107.6278458, 177.2025604, -90.6183243, 149.7760010, -255.2831116, 265.8214111

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5128497, upper bound: 86.5156336
time: 1.29 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143646, upper bound: 86.5145628
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5142873, upper bound: 86.5143266
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -48.3252106, 106.8312531, -40.5274315, 89.5396118, -136.8452148, 146.2014313
1: -97.1261520, 159.1792603, -82.2189789, 133.5822601, -230.3570099, 240.6932220
2: -75.3381729, 152.7682343, -63.5010147, 128.7380524, -203.6271515, 215.8849030
3: -114.7816086, 183.3250275, -96.5762939, 154.2429504, -268.2353821, 279.3317261
4: -106.3894730, 175.2429657, -89.2017517, 147.5167542, -251.8094482, 262.5410461

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5152294, upper bound: 86.5152294
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5152294, upper bound: 86.5152294
time: 0.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.35 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5151250, upper bound: 86.5154500
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5151250, upper bound: 86.5154501
NS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5114215, upper bound: 86.5122361
NS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5110740, upper bound: 86.5121682
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5151077, upper bound: 86.5151077
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5151077, upper bound: 86.5151077
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5112280, upper bound: 86.5119988
NS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5087855, upper bound: 86.5087855
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5158654, upper bound: 86.5165058
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5160456, upper bound: 86.5164353
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5158654, upper bound: 86.5165058
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5160456, upper bound: 86.5164353
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5177835, upper bound: 86.5167562
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5180195, upper bound: 86.5170243
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5158643, upper bound: 86.5160295
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5158643, upper bound: 86.5160532
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5176377, upper bound: 86.5181877
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5177355, upper bound: 86.5177355
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5176377, upper bound: 86.5181877
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5177355, upper bound: 86.5177355
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5169131, upper bound: 86.5169406
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5157348, upper bound: 86.5166083
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5169131, upper bound: 86.5169406
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5157348, upper bound: 86.5166083
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5154599, upper bound: 86.5150349
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5154599, upper bound: 86.5150413
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5154599, upper bound: 86.5149905
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5154599, upper bound: 86.5149788
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5143646, upper bound: 86.5145628
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5142873, upper bound: 86.5143266
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5152294, upper bound: 86.5152294
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -86.5152294, upper bound: 86.5152294

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -27.0296421, 59.8296585, -27.9589462, 61.9158974, -88.9455414, 87.7886047
1: -55.9732475, 89.3801804, -58.0258255, 92.5207672, -148.4940033, 147.4059601
2: -42.8463364, 86.9053955, -44.3505783, 90.0344009, -132.8807068, 131.2559814
3: -65.1015396, 103.7798309, -67.4060364, 107.4967422, -172.5982666, 171.1858673
4: -59.8141556, 99.0363770, -61.8487091, 102.5747604, -162.3889160, 160.8850861

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169792, upper bound: 86.5164804
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169792, upper bound: 86.5165862
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -31.3956661, 69.0581207, -27.9589462, 61.9158974, -93.3115616, 97.0170670
1: -65.6972733, 103.1429825, -58.0258255, 92.5207672, -158.2180328, 161.1687622
2: -50.0314598, 100.9354095, -44.3505783, 90.0344009, -140.0658417, 145.2859802
3: -75.9418259, 120.0693359, -67.4060364, 107.4967422, -183.4385681, 187.4753265
4: -69.3835678, 114.7670670, -61.8487091, 102.5747604, -171.9583282, 176.6157837

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169792, upper bound: 86.5164804
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169792, upper bound: 86.5165862
time: 1.22 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -27.1931400, 60.2039528, -27.7809429, 61.4963074, -88.6894455, 87.9848862
1: -56.2813339, 89.9903183, -57.6178703, 91.8910065, -148.1723328, 147.6081696
2: -43.0917091, 87.4708862, -44.0578232, 89.4094009, -132.5010986, 131.5287170
3: -65.4815598, 104.4893875, -66.9583282, 106.7554932, -172.2370605, 171.4477234
4: -60.1683617, 99.6779022, -61.4574432, 101.8600998, -162.0284424, 161.1353302

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169899, upper bound: 86.5161789
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169899, upper bound: 86.5161789
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -31.6714172, 69.6833954, -27.7809429, 61.4963074, -93.1677246, 97.4643402
1: -66.2517776, 104.1287155, -57.6178703, 91.8910065, -158.1427612, 161.7465363
2: -50.4514542, 101.8791504, -44.0578232, 89.4094009, -139.8608246, 145.9369812
3: -76.5985870, 121.2061920, -66.9583282, 106.7554932, -183.3540802, 188.1645203
4: -69.9875259, 115.8404999, -61.4574432, 101.8600998, -171.8475647, 177.2979431

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169899, upper bound: 86.5161789
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169899, upper bound: 86.5161788
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -28.6693382, 63.8680954, -36.1670074, 80.1096497, -108.7789917, 100.0351028
1: -59.4955025, 95.5737534, -74.1178131, 119.1594467, -178.6549072, 169.6915588
2: -45.4857712, 92.7499619, -56.9584007, 115.5875244, -161.0732880, 149.7083282
3: -69.1241074, 110.7785950, -86.6240311, 138.0152588, -207.1393585, 197.4026184
4: -63.4269066, 105.7987061, -79.7528152, 132.0178833, -195.4447937, 185.5514984

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5173176, upper bound: 86.5168634
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5176906, upper bound: 86.5172539
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -27.1096687, 60.0939064, -36.6140022, 81.0636826, -108.1733398, 96.7079086
1: -56.1009560, 89.7716064, -75.0314484, 120.5984192, -176.6993713, 164.8030548
2: -42.9600029, 87.2324142, -57.6588249, 116.9952393, -159.9552460, 144.8911896
3: -65.2728195, 104.2000809, -87.6985016, 139.6783905, -204.9512024, 191.8985291
4: -59.9840469, 99.4418640, -80.7347488, 133.6008759, -193.5849304, 180.1766052

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5181418, upper bound: 86.5171742
time: 1.35 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5181418, upper bound: 86.5171742
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -28.6693382, 63.8680954, -36.2798004, 80.3788071, -109.0481415, 100.1478958
1: -59.4955025, 95.5737534, -74.3319626, 119.6279984, -179.1234589, 169.9057007
2: -45.4857712, 92.7499619, -57.1269913, 116.0178833, -161.5036469, 149.8769226
3: -69.1241074, 110.7785950, -86.8872528, 138.5670776, -207.6911621, 197.6658478
4: -63.4269066, 105.7987061, -79.9966660, 132.4967804, -195.9236908, 185.7953186

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5150609, upper bound: 86.5163225
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5138546, upper bound: 86.5162030
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -27.1096687, 60.0939064, -36.7358246, 81.3510590, -108.4607086, 96.8297272
1: -56.1009560, 89.7716064, -75.2604599, 121.0923767, -177.1933136, 165.0320740
2: -42.9600029, 87.2324142, -57.8421402, 117.4501190, -160.4101257, 145.0745239
3: -65.2728195, 104.2000809, -87.9805527, 140.2649689, -205.5377808, 192.1806335
4: -59.9840469, 99.4418640, -80.9991684, 134.1023407, -194.0863647, 180.4410400

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5160456, upper bound: 86.5164353
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5160456, upper bound: 86.5164353
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -25.7764244, 57.3719559, -37.6875496, 83.8887024, -109.6651154, 95.0594864
1: -53.4198761, 85.7328720, -77.3891068, 125.0122375, -178.4321136, 163.1219635
2: -40.8708458, 83.2374878, -59.4013863, 121.0785904, -161.9494171, 142.6388550
3: -62.0850143, 99.5099182, -90.3688965, 144.7137604, -206.7987671, 189.8787994
4: -57.0365486, 94.9666672, -83.0848770, 138.3603821, -195.3969269, 178.0515289

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5177835, upper bound: 86.5164936
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5177835, upper bound: 86.5164936
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -27.7638302, 62.2445374, -37.6859741, 83.8638535, -111.6276855, 99.9305038
1: -57.6692200, 93.2213211, -77.3648453, 124.9647751, -182.6339874, 170.5861664
2: -44.0806656, 90.3852081, -59.3951340, 121.0426941, -165.1233521, 149.7803345
3: -66.9711761, 108.2842407, -90.3559341, 144.6732025, -211.6443787, 198.6401672
4: -61.4500122, 103.2467957, -83.0834274, 138.3260040, -199.7760162, 186.3302002

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5180195, upper bound: 86.5167874
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5177835, upper bound: 86.5167874
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -28.5968876, 63.7056198, -36.9628029, 81.8988342, -110.4957123, 100.6684189
1: -59.2851753, 95.3532104, -75.8259735, 121.8751602, -181.1603394, 171.1791687
2: -45.3462524, 92.4675446, -58.2292061, 118.2659454, -163.6121826, 150.6967468
3: -68.9229126, 110.4859924, -88.5776749, 141.2120667, -210.1349792, 199.0636597
4: -63.2563705, 105.5058136, -81.4928894, 135.0363770, -198.2927399, 186.9987030

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5158643, upper bound: 86.5160295
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5158643, upper bound: 86.5160295
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -27.2940578, 60.5126801, -36.9628029, 81.8988342, -109.1928787, 97.4754639
1: -56.4556313, 90.4447098, -75.8259735, 121.8751602, -178.3307648, 166.2706757
2: -43.2409782, 87.8608017, -58.2292061, 118.2659454, -161.5068665, 146.0899963
3: -65.7066040, 104.9757309, -88.5776749, 141.2120667, -206.9186554, 193.5534058
4: -60.3849869, 100.1557236, -81.4928894, 135.0363770, -195.4213409, 181.6486053

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5158643, upper bound: 86.5160532
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5158643, upper bound: 86.5160532
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -34.3631248, 75.9497833, -26.3027344, 58.0704346, -92.4335632, 102.2525177
1: -69.7310257, 113.0045547, -53.9150963, 86.7523270, -156.4833527, 166.9196472
2: -53.8758888, 109.0151901, -41.5236359, 83.8327179, -137.7086029, 150.5387878
3: -82.0784302, 130.3429718, -63.1482277, 100.1951675, -182.2735748, 193.4911804
4: -75.8278122, 124.6731491, -58.2591209, 95.7540970, -171.5819092, 182.9322510

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5178227, upper bound: 86.5184639
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5178227, upper bound: 86.5185667
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -34.4838486, 76.2522125, -26.1283360, 57.6654701, -92.1493073, 102.3805466
1: -69.9694748, 113.5075226, -53.5205650, 86.1466522, -156.1160736, 167.0280609
2: -54.0617714, 109.4839783, -41.2367783, 83.2327728, -137.2945404, 150.7207642
3: -82.3614655, 130.9402924, -62.7109642, 99.4847488, -181.8462219, 193.6512299
4: -76.0889969, 125.2055054, -57.8740845, 95.0703354, -171.1593323, 183.0795593

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5178873, upper bound: 86.5184639
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5178873, upper bound: 86.5185667
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -34.3631248, 75.9497833, -35.2760010, 77.9963226, -112.0758057, 110.9385071
1: -69.7310257, 113.0045547, -71.7408676, 116.0812225, -185.8122559, 184.7454071
2: -53.8758888, 109.0151901, -55.3609734, 112.0730743, -165.9489288, 164.3761444
3: -82.0784302, 130.3429718, -84.3318100, 133.9859467, -216.0643616, 214.6747742
4: -75.8278122, 124.6731491, -77.8325272, 128.1418762, -203.6387329, 202.1594086

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5176377, upper bound: 86.5176377
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5176377, upper bound: 86.5177355
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -34.4838486, 76.2522125, -35.0990410, 77.5870209, -111.7872391, 111.0664291
1: -69.9694748, 113.5075226, -71.3386765, 115.4714737, -185.4409485, 184.8461914
2: -54.0617714, 109.4839783, -55.0688667, 111.4684830, -165.5302582, 164.5528412
3: -82.3614655, 130.9402924, -83.8879395, 133.2706604, -215.6321259, 214.8282013
4: -76.0889969, 125.2055054, -77.4410553, 127.4504166, -203.2053375, 202.3192291

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5177355, upper bound: 86.5176377
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5177355, upper bound: 86.5177355
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -47.4946861, 104.7847672, -25.9378510, 57.2738800, -104.2416306, 130.2233276
1: -95.3192749, 155.9450531, -53.1330643, 85.5472870, -180.8665466, 208.7147675
2: -74.0260086, 149.6979980, -40.9368057, 82.6434555, -156.2342682, 190.6347961
3: -112.7068100, 179.5425568, -62.2574577, 98.7919540, -210.7889862, 241.8000183
4: -104.5622787, 171.7423248, -57.4569397, 94.4142914, -197.5808411, 228.7739563

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5163847, upper bound: 86.5148171
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5150405, upper bound: 86.5150463
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5150977, upper bound: 86.5160081
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -46.9167099, 103.5226212, -25.3287563, 56.0401268, -102.4543304, 128.3887939
1: -93.9498138, 154.1301727, -51.8347397, 83.7062912, -177.6560974, 205.6305847
2: -73.0222168, 147.7697906, -39.9509163, 80.7903137, -153.4149933, 187.7207031
3: -111.2460938, 177.3283997, -60.7699242, 96.6358643, -207.1952362, 238.0983124
4: -103.2473145, 169.6036377, -56.1106873, 92.3401794, -194.2987671, 225.3657532

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149905, upper bound: 86.5154133
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5150069, upper bound: 86.5160087
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -47.8912086, 105.7134781, -34.8846855, 77.1360779, -124.0262222, 139.6322632
1: -96.1951370, 157.4194031, -70.8945618, 114.7802505, -210.7030029, 227.5910034
2: -74.6553268, 151.1492157, -54.7283669, 110.7853394, -184.9056549, 205.7024689
3: -113.7032776, 181.3213501, -83.3725433, 132.4659119, -245.3358002, 264.3767700
4: -105.4338303, 173.3377838, -76.9690628, 126.6857605, -230.0829620, 248.8221130

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5162840, upper bound: 86.5145153
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5150349, upper bound: 86.5154599
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5150413, upper bound: 86.5157780
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -47.3046112, 104.4229431, -34.3422165, 76.0354385, -122.3524017, 137.8235779
1: -94.7954483, 155.5502777, -69.7409821, 113.1333618, -207.6560211, 224.5810089
2: -73.6360703, 149.1661835, -53.8469543, 109.1300507, -182.2556915, 202.8472290
3: -112.2144928, 179.0443268, -82.0537033, 130.5261383, -241.9218903, 260.7777100
4: -104.0994186, 171.1452942, -75.7649002, 124.8354263, -226.9881439, 245.4809113

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149905, upper bound: 86.5155631
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149788, upper bound: 86.5157751
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -35.4599266, 78.8901978, -40.2767220, 89.0018616, -124.1023865, 118.5915833
1: -72.1557770, 117.6875381, -81.9030914, 132.6816406, -204.8373718, 199.5905914
2: -55.6568451, 113.3311996, -63.2064323, 128.0350952, -183.6919403, 176.3572998
3: -84.8176117, 135.7032471, -96.0505676, 153.2986450, -238.1162567, 231.4863892
4: -78.2330933, 129.7001038, -88.7001419, 146.6672974, -224.3628235, 217.3396606

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149496, upper bound: 86.5146475
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5129167, upper bound: 86.5145044
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -34.2925682, 75.9090881, -40.7565727, 89.9863358, -123.9563751, 116.1763077
1: -69.6418610, 112.9513474, -82.8646011, 134.1596069, -203.8014526, 195.8159027
2: -53.7763557, 108.9736176, -63.9484558, 129.4923706, -183.2687225, 172.9202881
3: -81.9415131, 130.3253174, -97.2004395, 155.0262604, -236.9677429, 227.4868011
4: -75.6605759, 124.6303711, -89.7573471, 148.2807465, -223.4974670, 213.5269928

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149572, upper bound: 86.5139792
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157780, upper bound: 86.5150413
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -34.9142876, 77.7816162, -39.6093903, 87.5564041, -122.1347885, 116.8313675
1: -70.9907303, 116.0253143, -80.3451767, 130.5781708, -201.5689087, 196.3651733
2: -54.7688484, 111.6660614, -62.0616608, 125.8443146, -180.6131592, 173.5843353
3: -83.4891663, 133.7431335, -94.3667374, 150.7711182, -234.2602844, 227.8594971
4: -77.0222092, 127.8370361, -87.1837463, 144.2325745, -220.7604370, 214.0586395

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149496, upper bound: 86.5146056
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151559, upper bound: 86.5145019
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -33.7523346, 74.8141327, -40.1085014, 88.5803833, -122.0374069, 114.4495850
1: -68.4929352, 111.3106232, -81.3536758, 132.1220398, -200.6149750, 192.6643066
2: -52.8990059, 107.3260803, -62.8370895, 127.3652115, -180.2642212, 170.1631470
3: -80.6281738, 128.3930664, -95.5653458, 152.5733948, -233.2015686, 223.9345245
4: -74.4618149, 122.7913513, -88.2844086, 145.9198608, -219.9841766, 210.3118896

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157751, upper bound: 86.5149788
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157751, upper bound: 86.5149788
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -48.9452972, 108.6901627, -40.4555969, 89.4373093, -137.3099823, 147.8230286
1: -98.5202103, 162.2587128, -82.2601776, 133.3589935, -231.5072479, 243.5566711
2: -76.3632812, 155.5211182, -63.4861908, 128.6416473, -204.5394440, 218.3558807
3: -116.3477402, 186.7589111, -96.4799271, 154.0591431, -269.5828247, 282.2930298
4: -107.7776413, 178.4667969, -89.0889206, 147.3843536, -252.9543304, 265.2257385

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5137475, upper bound: 86.5141628
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5140446, upper bound: 86.5140505
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -48.3466988, 106.8648834, -40.9661560, 90.4966354, -137.8017273, 146.6345825
1: -97.1714554, 159.2430725, -83.2828522, 134.9533234, -231.7608185, 241.7687378
2: -75.3781586, 152.8537598, -64.2761612, 130.2035522, -205.1305695, 216.6664581
3: -114.8389359, 183.4018250, -97.7033157, 155.9174347, -269.9694824, 280.4496460
4: -106.4433746, 175.3153076, -90.2128067, 149.1217499, -253.4382782, 263.4694824

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5142873, upper bound: 86.5143266
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5142873, upper bound: 86.5143266
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -48.1808662, 106.4197159, -40.5274315, 89.5396118, -136.6818085, 145.7785339
1: -96.7968292, 158.5176239, -82.2189789, 133.5822601, -230.0093994, 240.0252228
2: -75.1119080, 152.1670227, -63.5010147, 128.7380524, -203.3692627, 215.2710876
3: -114.4108047, 182.5747528, -96.5762939, 154.2429504, -267.8324890, 278.5704651
4: -106.0686264, 174.5245667, -89.2017517, 147.5167542, -251.4152832, 261.7830505

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145229, upper bound: 86.5143646
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5142631, upper bound: 86.5142631
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -47.6190529, 105.1861191, -40.5274315, 89.5396118, -136.1361847, 144.5608673
1: -95.4436264, 156.7297668, -82.2189789, 133.5822601, -228.6618500, 238.2337646
2: -74.1301727, 150.2608185, -63.5010147, 128.7380524, -202.4164276, 213.3588409
3: -112.9793320, 180.3897095, -96.5762939, 154.2429504, -266.4137878, 276.3658142
4: -104.7869263, 172.4225769, -89.2017517, 147.5167542, -250.2302399, 259.7115784

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145229, upper bound: 86.5144108
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5142631, upper bound: 86.5142631
time: 0.73 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.38 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5169792, upper bound: 86.5164804
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5169792, upper bound: 86.5165862
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5169792, upper bound: 86.5164804
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5169792, upper bound: 86.5165862
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5169899, upper bound: 86.5161789
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5169899, upper bound: 86.5161789
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5169899, upper bound: 86.5161789
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5169899, upper bound: 86.5161788
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5173176, upper bound: 86.5168634
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5176906, upper bound: 86.5172539
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5181418, upper bound: 86.5171742
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5181418, upper bound: 86.5171742
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5150609, upper bound: 86.5163225
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5138546, upper bound: 86.5162030
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5160456, upper bound: 86.5164353
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5160456, upper bound: 86.5164353
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5177835, upper bound: 86.5164936
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5177835, upper bound: 86.5164936
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5180195, upper bound: 86.5167874
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5177835, upper bound: 86.5167874
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5158643, upper bound: 86.5160295
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5158643, upper bound: 86.5160295
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5158643, upper bound: 86.5160532
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5158643, upper bound: 86.5160532
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5178227, upper bound: 86.5184639
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5178227, upper bound: 86.5185667
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5178873, upper bound: 86.5184639
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5178873, upper bound: 86.5185667
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5176377, upper bound: 86.5176377
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5176377, upper bound: 86.5177355
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5177355, upper bound: 86.5176377
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5177355, upper bound: 86.5177355
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5150405, upper bound: 86.5150463
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5150977, upper bound: 86.5160081
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5149905, upper bound: 86.5154133
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5150069, upper bound: 86.5160087
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5150349, upper bound: 86.5154599
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5150413, upper bound: 86.5157780
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5149905, upper bound: 86.5155631
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5149788, upper bound: 86.5157751
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5149496, upper bound: 86.5146475
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5129167, upper bound: 86.5145044
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5149572, upper bound: 86.5139792
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5157780, upper bound: 86.5150413
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5149496, upper bound: 86.5146056
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5151559, upper bound: 86.5145019
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5157751, upper bound: 86.5149788
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5157751, upper bound: 86.5149788
NS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5137475, upper bound: 86.5141628
NS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5140446, upper bound: 86.5140505
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5142873, upper bound: 86.5143266
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5142873, upper bound: 86.5143266
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5145229, upper bound: 86.5143646
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5142631, upper bound: 86.5142631
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5145229, upper bound: 86.5144108
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -86.5142631, upper bound: 86.5142631

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -27.0296421, 59.8296585, -27.0296421, 59.8296585, -86.8592987, 86.8592987
1: -55.9732475, 89.3801804, -55.9732475, 89.3801804, -145.3533783, 145.3533783
2: -42.8463364, 86.9053955, -42.8463364, 86.9053955, -129.7517395, 129.7517395
3: -65.1015396, 103.7798309, -65.1015396, 103.7798309, -168.8813782, 168.8813782
4: -59.8141556, 99.0363770, -59.8141556, 99.0363770, -158.8505249, 158.8505249

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5182752, upper bound: 86.5190413
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5182752, upper bound: 86.5191615
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -27.0296421, 59.8296585, -27.1931400, 60.2039528, -87.2335968, 87.0227966
1: -55.9732475, 89.3801804, -56.2813339, 89.9903183, -145.9635620, 145.6614685
2: -42.8463364, 86.9053955, -43.0917091, 87.4708862, -130.3172302, 129.9971008
3: -65.1015396, 103.7798309, -65.4815598, 104.4893875, -169.5909119, 169.2613831
4: -59.8141556, 99.0363770, -60.1683617, 99.6779022, -159.4920502, 159.2047272

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5182752, upper bound: 86.5192508
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5182752, upper bound: 86.5195506
time: 1.21 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -31.3956661, 69.0581207, -27.0296421, 59.8296585, -91.2253265, 96.0877609
1: -65.6972733, 103.1429825, -55.9732475, 89.3801804, -155.0774078, 159.1161804
2: -50.0314598, 100.9354095, -42.8463364, 86.9053955, -136.9368591, 143.7817383
3: -75.9418259, 120.0693359, -65.1015396, 103.7798309, -179.7216492, 185.1708527
4: -69.3835678, 114.7670670, -59.8141556, 99.0363770, -168.4199219, 174.5812225

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5113671, upper bound: 86.5122348
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5110740, upper bound: 86.5121665
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -31.3956661, 69.0581207, -27.1931400, 60.2039528, -91.5996170, 96.2512589
1: -65.6972733, 103.1429825, -56.2813339, 89.9903183, -155.6875916, 159.4242859
2: -50.0314598, 100.9354095, -43.0917091, 87.4708862, -137.5023499, 144.0271149
3: -75.9418259, 120.0693359, -65.4815598, 104.4893875, -180.4312134, 185.5509033
4: -69.3835678, 114.7670670, -60.1683617, 99.6779022, -169.0614319, 174.9354248

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5113671, upper bound: 86.5122348
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5110740, upper bound: 86.5121665
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -27.1931400, 60.2039528, -27.0296421, 59.8296585, -87.0227966, 87.2335968
1: -56.2813339, 89.9903183, -55.9732475, 89.3801804, -145.6614685, 145.9635620
2: -43.0917091, 87.4708862, -42.8463364, 86.9053955, -129.9971008, 130.3172302
3: -65.4815598, 104.4893875, -65.1015396, 103.7798309, -169.2613831, 169.5909119
4: -60.1683617, 99.6779022, -59.8141556, 99.0363770, -159.2047272, 159.4920502

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5185123, upper bound: 86.5187404
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5194164, upper bound: 86.5190304
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -27.1931400, 60.2039528, -27.1931400, 60.2039528, -87.3970947, 87.3970947
1: -56.2813339, 89.9903183, -56.2813339, 89.9903183, -146.2716522, 146.2716522
2: -43.0917091, 87.4708862, -43.0917091, 87.4708862, -130.5625916, 130.5625916
3: -65.4815598, 104.4893875, -65.4815598, 104.4893875, -169.9709473, 169.9709473
4: -60.1683617, 99.6779022, -60.1683617, 99.6779022, -159.8462372, 159.8462372

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5185123, upper bound: 86.5187555
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5194164, upper bound: 86.5191355
time: 1.24 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -31.6714172, 69.6833954, -27.0296421, 59.8296585, -91.5010757, 96.7130356
1: -66.2517776, 104.1287155, -55.9732475, 89.3801804, -155.6318970, 160.1019287
2: -50.4514542, 101.8791504, -42.8463364, 86.9053955, -137.3568115, 144.7254791
3: -76.5985870, 121.2061920, -65.1015396, 103.7798309, -180.3784180, 186.3077393
4: -69.9875259, 115.8404999, -59.8141556, 99.0363770, -169.0238800, 175.6546631

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5112280, upper bound: 86.5119972
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5087855, upper bound: 86.5087855
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -31.6714172, 69.6833954, -27.1931400, 60.2039528, -91.8753662, 96.8765335
1: -66.2517776, 104.1287155, -56.2813339, 89.9903183, -156.2420959, 160.4100342
2: -50.4514542, 101.8791504, -43.0917091, 87.4708862, -137.9223328, 144.9708557
3: -76.5985870, 121.2061920, -65.4815598, 104.4893875, -181.0879822, 186.6877441
4: -69.9875259, 115.8404999, -60.1683617, 99.6779022, -169.6653900, 176.0088654

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5112280, upper bound: 86.5119972
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5087855, upper bound: 86.5087855
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -27.8640976, 61.8685036, -33.1510010, 73.1881256, -101.0522232, 95.0195007
1: -57.7033501, 92.5452271, -67.5838165, 108.7436676, -166.4470215, 160.1290436
2: -44.1674690, 89.8760605, -52.0604744, 105.4869919, -149.6544647, 141.9365234
3: -67.1221390, 107.2684174, -79.2040253, 125.8692093, -192.9913330, 186.4724426
4: -61.6425438, 102.4509735, -73.0597153, 120.4650650, -182.1075897, 175.5106812

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169338, upper bound: 86.5163088
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5166372, upper bound: 86.5163359
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -28.5923843, 63.6902809, -35.9435692, 79.5903091, -108.1826935, 99.6338425
1: -59.3181534, 95.3040085, -73.6066132, 118.3656006, -177.6837311, 168.9106140
2: -45.3576241, 92.4783020, -56.5862083, 114.7902985, -160.1479187, 149.0645142
3: -68.9288483, 110.4584503, -86.0582962, 137.0550537, -205.9838715, 196.5167542
4: -63.2563438, 105.4919891, -79.2577057, 131.1193085, -194.3756104, 184.7496948

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5174089, upper bound: 86.5165843
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5165611, upper bound: 86.5163827
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -27.1096687, 60.0939064, -37.4429932, 83.2979965, -110.4076462, 97.5368958
1: -56.1009560, 89.7716064, -76.7873306, 124.1328278, -180.2337494, 166.5589294
2: -42.9600029, 87.2324142, -58.9865570, 120.1814346, -163.1414337, 146.2189484
3: -65.2728195, 104.2000809, -89.7383881, 143.6556244, -208.9284363, 193.9384613
4: -59.9840469, 99.4418640, -82.5582275, 137.3408203, -197.3248596, 182.0000916

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5179034, upper bound: 86.5165036
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5171267, upper bound: 86.5164392
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -27.1096687, 60.0939064, -36.2201385, 80.2382202, -107.3478699, 96.3140411
1: -56.1009560, 89.7716064, -74.1918640, 119.3597183, -175.4606781, 163.9634705
2: -42.9600029, 87.2324142, -57.0248795, 115.7748947, -158.7348938, 144.2572632
3: -65.2728195, 104.2000809, -86.7421799, 138.2268982, -203.4997253, 190.9422607
4: -59.9840469, 99.4418640, -79.8645859, 132.2126770, -192.1967163, 179.3063965

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5179034, upper bound: 86.5165036
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5171267, upper bound: 86.5164392
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -27.8640976, 61.8685036, -33.4920731, 73.9373932, -101.8014832, 95.3605804
1: -57.7033501, 92.5452271, -68.2814407, 109.9154358, -167.6187897, 160.8266602
2: -44.1674690, 89.8760605, -52.6077919, 106.6270599, -150.7945251, 142.4838562
3: -67.1221390, 107.2684174, -80.0148697, 127.2453995, -194.3675232, 187.2832947
4: -61.6425438, 102.4509735, -73.8056107, 121.7455368, -183.3880768, 176.2565765

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5131244, upper bound: 86.5151342
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5131624, upper bound: 86.5152562
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -28.5923843, 63.6902809, -36.0564423, 79.8585358, -108.4509125, 99.7467194
1: -59.3181534, 95.3040085, -73.8144073, 118.8341675, -178.1523132, 169.1184082
2: -45.3576241, 92.4783020, -56.7529182, 115.2225189, -160.5801392, 149.2312164
3: -68.9288483, 110.4584503, -86.3179169, 137.6053925, -206.5342102, 196.7763519
4: -63.2563438, 105.4919891, -79.5011902, 131.5942993, -194.8506012, 184.9931641

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5137950, upper bound: 86.5161757
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5137950, upper bound: 86.5162030
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -27.1096687, 60.0939064, -37.5696564, 83.5958710, -110.7055206, 97.6635590
1: -56.1009560, 89.7716064, -77.0162277, 124.6342926, -180.7352295, 166.7878418
2: -42.9600029, 87.2324142, -59.1699524, 120.6610794, -163.6210785, 146.4023438
3: -65.2728195, 104.2000809, -90.0297699, 144.2353973, -209.5081940, 194.2298431
4: -59.9840469, 99.4418640, -82.8281326, 137.8683777, -197.8524170, 182.2699890

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151408, upper bound: 86.5153017
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5152689, upper bound: 86.5152916
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -27.1096687, 60.0939064, -36.3439560, 80.5253525, -107.6350098, 96.4378662
1: -56.1009560, 89.7716064, -74.4211197, 119.8519669, -175.9529266, 164.1927185
2: -42.9600029, 87.2324142, -57.2110786, 116.2276077, -159.1876068, 144.4434509
3: -65.2728195, 104.2000809, -87.0265121, 138.8104248, -204.0832520, 191.2265930
4: -59.9840469, 99.4418640, -80.1328888, 132.7107391, -192.6947937, 179.5747375

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151408, upper bound: 86.5153016
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151408, upper bound: 86.5152916
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -25.7764244, 57.3719559, -37.0138931, 82.3858871, -108.1623001, 94.3858490
1: -53.4198761, 85.7328720, -75.9018860, 122.7563477, -176.1762085, 161.6347656
2: -40.8708458, 83.2374878, -58.3063850, 118.8230972, -159.6939087, 141.5438538
3: -62.0850143, 99.5099182, -88.7042770, 142.0440674, -204.1290894, 188.2141876
4: -57.0365486, 94.9666672, -81.6085892, 135.8098145, -192.8463593, 176.5752106

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5177835, upper bound: 86.5164936
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5177835, upper bound: 86.5164936
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -25.7764244, 57.3719559, -37.1226807, 82.6482925, -108.4246979, 94.4946365
1: -53.4198761, 85.7328720, -76.0933380, 123.2080002, -176.6278687, 161.8261871
2: -40.8708458, 83.2374878, -58.4607544, 119.2538605, -160.1247101, 141.6982422
3: -62.0850143, 99.5099182, -88.9524155, 142.5615082, -204.6465149, 188.4623108
4: -57.0365486, 94.9666672, -81.8390350, 136.2774048, -193.3139496, 176.8056641

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5177835, upper bound: 86.5164936
time: 1.10 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5177835, upper bound: 86.5164936
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -27.7638302, 62.2445374, -36.9197769, 82.1623459, -109.9261780, 99.1643143
1: -57.6692200, 93.2213211, -75.6833191, 122.4114914, -180.0806732, 168.9046326
2: -44.0806656, 90.3852081, -58.1524544, 118.4932327, -162.5738983, 148.5376587
3: -66.9711761, 108.2842407, -88.4665527, 141.6558685, -208.6270447, 196.7507935
4: -61.4500122, 103.2467957, -81.4035568, 135.4365692, -196.8865814, 184.6503448

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5180195, upper bound: 86.5167874
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5180195, upper bound: 86.5167874
time: 1.15 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -27.7638302, 62.2445374, -37.0402412, 82.4403687, -110.2042007, 99.2847748
1: -57.6692200, 93.2213211, -75.8947754, 122.8831024, -180.5523224, 169.1160889
2: -44.0806656, 90.3852081, -58.3248405, 118.9364014, -163.0170593, 148.7100372
3: -66.9711761, 108.2842407, -88.7402115, 142.1879425, -209.1591187, 197.0244446
4: -61.4500122, 103.2467957, -81.6596756, 135.9204559, -197.3704681, 184.9064636

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5180195, upper bound: 86.5167874
time: 1.10 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5177835, upper bound: 86.5167874
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -28.5968876, 63.7056198, -36.2201385, 80.2382202, -108.8350983, 99.9257584
1: -59.2851753, 95.3532104, -74.1918640, 119.3597183, -178.6448975, 169.5450745
2: -45.3462524, 92.4675446, -57.0248795, 115.7748947, -161.1211395, 149.4924316
3: -68.9229126, 110.4859924, -86.7421799, 138.2268982, -207.1498108, 197.2281647
4: -63.2563705, 105.5058136, -79.8645859, 132.2126770, -195.4690247, 185.3703613

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5138234, upper bound: 86.5156012
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5136932, upper bound: 86.5150274
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -28.5968876, 63.7056198, -36.3439560, 80.5253525, -109.1222382, 100.0495758
1: -59.2851753, 95.3532104, -74.4211197, 119.8519669, -179.1371460, 169.7743073
2: -45.3462524, 92.4675446, -57.2110786, 116.2276077, -161.5738525, 149.6786194
3: -68.9229126, 110.4859924, -87.0265121, 138.8104248, -207.7333374, 197.5125122
4: -63.2563705, 105.5058136, -80.1328888, 132.7107391, -195.9670868, 185.6387024

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5138234, upper bound: 86.5156012
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5136932, upper bound: 86.5150274
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -27.2940578, 60.5126801, -36.2201385, 80.2382202, -107.5322647, 96.7328110
1: -56.4556313, 90.4447098, -74.1918640, 119.3597183, -175.8153534, 164.6365662
2: -43.2409782, 87.8608017, -57.0248795, 115.7748947, -159.0158234, 144.8856812
3: -65.7066040, 104.9757309, -86.7421799, 138.2268982, -203.9334869, 191.7178955
4: -60.3849869, 100.1557236, -79.8645859, 132.2126770, -192.5976257, 180.0202789

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151419, upper bound: 86.5143892
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5153031, upper bound: 86.5148032
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -27.2940578, 60.5126801, -36.3439560, 80.5253525, -107.8194122, 96.8566208
1: -56.4556313, 90.4447098, -74.4211197, 119.8519669, -176.3076019, 164.8658142
2: -43.2409782, 87.8608017, -57.2110786, 116.2276077, -159.4685364, 145.0718536
3: -65.7066040, 104.9757309, -87.0265121, 138.8104248, -204.5170135, 192.0022430
4: -60.3849869, 100.1557236, -80.1328888, 132.7107391, -193.0956879, 180.2886047

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151419, upper bound: 86.5143892
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5153031, upper bound: 86.5148032
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -34.3631248, 75.9497833, -25.3807564, 56.0091057, -90.3722305, 101.3305359
1: -69.7310257, 113.0045547, -51.8998260, 83.6548004, -153.3858337, 164.9043732
2: -53.8758888, 109.0151901, -40.0329132, 80.7558899, -134.6317596, 149.0480957
3: -82.0784302, 130.3429718, -60.8711090, 96.5507126, -178.6291351, 191.2140503
4: -75.8278122, 124.6731491, -56.2363472, 92.2710190, -168.0987854, 180.9094849

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5171679, upper bound: 86.5184784
time: 1.26 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5162871, upper bound: 86.5179194
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -34.3631248, 75.9497833, -25.5914116, 56.5158539, -90.8789825, 101.5411987
1: -69.7310257, 113.0045547, -52.3210144, 84.4602203, -154.1912537, 165.3255615
2: -53.8758888, 109.0151901, -40.3567848, 81.5073318, -135.3832245, 149.3719482
3: -82.0784302, 130.3429718, -61.3729439, 97.4719849, -179.5503845, 191.7159119
4: -75.8278122, 124.6731491, -56.6925659, 93.1374130, -168.9651947, 181.3657227

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5171679, upper bound: 86.5184927
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5162871, upper bound: 86.5179194
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -34.4838486, 76.2522125, -25.3807564, 56.0091057, -90.4929428, 101.6329651
1: -69.9694748, 113.5075226, -51.8998260, 83.6548004, -153.6242676, 165.4073181
2: -54.0617714, 109.4839783, -40.0329132, 80.7558899, -134.8176575, 149.5168762
3: -82.3614655, 130.9402924, -60.8711090, 96.5507126, -178.9121704, 191.8113556
4: -76.0889969, 125.2055054, -56.2363472, 92.2710190, -168.3600006, 181.4418488

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5172348, upper bound: 86.5180476
time: 1.51 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5159458, upper bound: 86.5156441
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -34.4838486, 76.2522125, -25.5914116, 56.5158539, -90.9996872, 101.8436203
1: -69.9694748, 113.5075226, -52.3210144, 84.4602203, -154.4296722, 165.8285065
2: -54.0617714, 109.4839783, -40.3567848, 81.5073318, -135.5691071, 149.8407288
3: -82.3614655, 130.9402924, -61.3729439, 97.4719849, -179.8334503, 192.3132172
4: -76.0889969, 125.2055054, -56.6925659, 93.1374130, -169.2264099, 181.8980713

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5172348, upper bound: 86.5180476
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5159458, upper bound: 86.5156441
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -34.3631248, 75.9497833, -34.3631248, 75.9497833, -110.0216599, 110.0216599
1: -69.7310257, 113.0045547, -69.7310257, 113.0045547, -182.7355804, 182.7355804
2: -53.8758888, 109.0151901, -53.8758888, 109.0151901, -162.8910675, 162.8910675
3: -82.0784302, 130.3429718, -82.0784302, 130.3429718, -212.4213715, 212.4213715
4: -75.8278122, 124.6731491, -75.8278122, 124.6731491, -200.1460114, 200.1460114

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5147956, upper bound: 86.5163528
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143758, upper bound: 86.5161502
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -34.3631248, 75.9497833, -34.4838486, 76.2522125, -110.3276672, 110.1429901
1: -69.7310257, 113.0045547, -69.9694748, 113.5075226, -183.2385254, 182.9739990
2: -53.8758888, 109.0151901, -54.0617714, 109.4839783, -163.3598480, 163.0769653
3: -82.0784302, 130.3429718, -82.3614655, 130.9402924, -213.0186768, 212.7044373
4: -75.8278122, 124.6731491, -76.0889969, 125.2055054, -200.6992035, 200.4053345

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5147956, upper bound: 86.5163528
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143758, upper bound: 86.5161503
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -34.4838486, 76.2522125, -34.3631248, 75.9497833, -110.1429901, 110.3276672
1: -69.9694748, 113.5075226, -69.7310257, 113.0045547, -182.9739990, 183.2385254
2: -54.0617714, 109.4839783, -53.8758888, 109.0151901, -163.0769653, 163.3598480
3: -82.3614655, 130.9402924, -82.0784302, 130.3429718, -212.7044373, 213.0186920
4: -76.0889969, 125.2055054, -75.8278122, 124.6731491, -200.4053345, 200.6992035

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148357, upper bound: 86.5157709
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5142199, upper bound: 86.5142199
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -34.4838486, 76.2522125, -34.4838486, 76.2522125, -110.4489975, 110.4489975
1: -69.9694748, 113.5075226, -69.9694748, 113.5075226, -183.4769440, 183.4769440
2: -54.0617714, 109.4839783, -54.0617714, 109.4839783, -163.5457458, 163.5457458
3: -82.3614655, 130.9402924, -82.3614655, 130.9402924, -213.3017426, 213.3017426
4: -76.0889969, 125.2055054, -76.0889969, 125.2055054, -200.9588318, 200.9588318

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148357, upper bound: 86.5157709
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5142199, upper bound: 86.5142199
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -46.7191620, 103.1682892, -26.9579716, 59.9462662, -106.1010895, 129.5990295
1: -93.7476654, 153.5178986, -55.4043007, 89.8279266, -183.5755463, 208.5414886
2: -72.8183517, 147.3142090, -42.5958977, 86.6156311, -158.8764954, 189.9101105
3: -110.8516541, 176.7043152, -64.8163452, 103.6682510, -213.6370087, 241.5206299
4: -102.8574066, 169.0662842, -59.7022285, 98.9925690, -200.3625336, 228.2760162

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5146508, upper bound: 86.5146326
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143536, upper bound: 86.5129167
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -47.2565079, 104.2665329, -25.3650932, 56.0622635, -102.7953796, 129.1349640
1: -94.8134537, 155.1673431, -51.9035645, 83.7344131, -178.5478668, 206.6994019
2: -73.6424713, 148.9405670, -40.0145187, 80.8451004, -154.0333252, 188.9550781
3: -112.1299438, 178.6390228, -60.8587837, 96.6688385, -208.0654449, 239.4977875
4: -104.0385742, 170.8752594, -56.1913757, 92.3812408, -195.0275726, 226.6499023

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5140549, upper bound: 86.5151959
time: 1.25 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5150977, upper bound: 86.5160081
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -46.1200638, 101.8604813, -26.0806179, 58.1183205, -103.6953049, 127.4496994
1: -92.3308868, 151.6295624, -53.5350761, 87.0797195, -179.4106140, 204.8152313
2: -71.7793045, 145.3173676, -41.1794472, 83.8442001, -155.1205902, 186.4968109
3: -109.3389511, 174.4053345, -62.6780739, 100.4435272, -208.9094696, 237.0834045
4: -101.4947510, 166.8453522, -57.7634125, 95.9025726, -196.0078583, 224.1959076

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5146088, upper bound: 86.5149642
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143696, upper bound: 86.5149997
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -46.6804771, 103.0081787, -24.7680359, 54.8550110, -101.0365372, 127.3161240
1: -93.4500961, 153.3590240, -50.6298599, 81.9321747, -175.3822632, 203.6438904
2: -72.6423798, 147.0188446, -39.0465775, 79.0303574, -151.2542572, 186.0654144
3: -110.6748352, 176.4323730, -59.4006538, 94.5548706, -204.5188904, 235.8330231
4: -102.7277527, 168.7434082, -54.8715210, 90.3517075, -191.7942047, 223.2753601

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5146358, upper bound: 86.5152448
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145513, upper bound: 86.5155503
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -47.1545143, 104.1728287, -35.4599266, 78.8901978, -124.9416428, 138.6229248
1: -94.6882095, 155.1039276, -72.1557770, 117.6875381, -211.9239502, 226.4941711
2: -73.5037231, 148.8717499, -55.6568451, 113.3311996, -186.0762024, 204.3186951
3: -111.9336243, 178.6122131, -84.8176117, 135.7032471, -246.5234222, 263.0578918
4: -103.8137207, 170.7787476, -78.2330933, 129.7001038, -231.2401886, 247.4207458

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5146475, upper bound: 86.5149496
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145044, upper bound: 86.5150606
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -47.6655121, 105.2224045, -34.2925682, 75.9090881, -122.5614853, 138.5451660
1: -95.7133102, 156.6822205, -69.6418610, 112.9513474, -208.3581085, 225.5837708
2: -74.2908401, 150.4300842, -53.7763557, 108.9736176, -182.6887360, 204.0231171
3: -113.1554108, 180.4644775, -81.9415131, 130.3253174, -242.5988770, 262.0747681
4: -104.9374390, 172.5145569, -75.6605759, 124.6303711, -227.5039062, 246.6863403

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5139792, upper bound: 86.5149572
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5150413, upper bound: 86.5157780
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -46.5479965, 102.8393555, -34.9142876, 77.7816162, -123.2391815, 136.7639923
1: -93.2419281, 153.1655579, -70.9907303, 116.0253143, -208.8157196, 223.4040375
2: -72.4503021, 146.8235931, -54.7688484, 111.6660614, -183.3871765, 201.3912354
3: -110.3950500, 176.2603607, -83.4891663, 133.7431335, -243.0414276, 259.3731384
4: -102.4342422, 168.5072632, -77.0222092, 127.8370361, -228.0891571, 243.9907837

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5146056, upper bound: 86.5149807
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145019, upper bound: 86.5151559
time: 1.26 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -47.0816383, 103.9371643, -33.7523346, 74.8141327, -120.8960037, 136.7433472
1: -94.3207626, 154.8218231, -68.4929352, 111.3106232, -205.3243866, 222.5850525
2: -73.2762756, 148.4554596, -52.8990059, 107.3260803, -180.0508575, 201.1793213
3: -111.6739120, 178.1987915, -80.6281738, 128.3930664, -239.1997681, 258.4911194
4: -103.6086578, 170.3314209, -74.4618149, 122.7913513, -224.4258881, 243.3586273

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145976, upper bound: 86.5150405
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145226, upper bound: 86.5153661
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -35.4599266, 78.8901978, -39.8556213, 88.0735168, -123.1543427, 118.1668396
1: -72.1557770, 117.6875381, -80.9628677, 131.3029785, -203.4580536, 198.6291656
2: -55.6568451, 113.3311996, -62.5146484, 126.6532211, -182.3100281, 175.6526642
3: -84.8176117, 135.7032471, -95.0088730, 151.6697388, -236.4873505, 230.4326324
4: -78.2330933, 129.7001038, -87.7713547, 145.0857849, -222.7415619, 216.4032288

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148232, upper bound: 86.5144582
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148232, upper bound: 86.5144988
time: 1.06 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -35.3968506, 78.7458115, -39.7908630, 87.9083710, -122.9612808, 117.9620361
1: -72.0172882, 117.4726715, -80.8539963, 131.0765533, -203.0938416, 198.3266449
2: -55.5539932, 113.1200180, -62.4060669, 126.4743652, -182.0283508, 175.3638916
3: -84.6608429, 135.4521637, -94.8500519, 151.4444275, -236.1052704, 230.0517120
4: -78.0953293, 129.4576569, -87.6442337, 144.8674927, -222.4511871, 216.0386963

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148283, upper bound: 86.5144583
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148283, upper bound: 86.5145044
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -33.5070457, 73.9518204, -37.4817429, 82.5800323, -115.7310028, 110.9139709
1: -67.8817749, 110.0183411, -75.7397232, 123.0874481, -190.9212799, 185.7580566
2: -52.4845848, 106.1573334, -58.6318893, 118.6599503, -171.1445312, 164.7045441
3: -79.9781570, 126.9108429, -89.1416550, 142.1144867, -222.0926361, 215.9073486
4: -73.9183350, 121.3607712, -82.5210800, 135.9022522, -209.2826691, 202.9312134

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5144810, upper bound: 86.5133047
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5146306, upper bound: 86.5135148
time: 1.20 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -34.2075539, 75.7134476, -40.4389458, 89.2465820, -123.1246796, 115.6500626
1: -69.4487000, 112.6548691, -82.1593170, 133.0320740, -202.4807739, 194.8141785
2: -53.6348419, 108.6743622, -63.4284821, 128.3945160, -182.0293579, 172.0629425
3: -81.7270279, 129.9663696, -96.4104156, 153.7099609, -235.4369812, 226.2983398
4: -75.4722366, 124.2939148, -89.0588379, 147.0093842, -222.0176849, 212.4259338

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5153348, upper bound: 86.5144944
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5153762, upper bound: 86.5145260
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -34.9142876, 77.7816162, -39.1811714, 86.6110611, -121.1682281, 116.3992920
1: -70.9907303, 116.0253143, -79.3871307, 129.1705322, -200.1612549, 195.3814240
2: -54.7688484, 111.6660614, -61.3582993, 124.4381332, -179.2069855, 172.8672028
3: -83.4891663, 133.7431335, -93.3062134, 149.1088104, -232.5979767, 226.7859802
4: -77.0222092, 127.8370361, -86.2413406, 142.6230164, -219.1102753, 213.1082916

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148668, upper bound: 86.5144581
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148668, upper bound: 86.5144930
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -34.8509102, 77.6366730, -39.1390610, 86.4901581, -121.0274048, 116.2178116
1: -70.8515930, 115.8094711, -79.3293991, 129.0242767, -199.8758240, 195.1388397
2: -54.6655388, 111.4539032, -61.2856903, 124.3340988, -178.9996338, 172.6140289
3: -83.3316956, 133.4906006, -93.2040787, 148.9737244, -232.3054047, 226.4631958
4: -76.8837967, 127.5942154, -86.1616287, 142.4906158, -218.9082336, 212.7925110

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5150318, upper bound: 86.5144593
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5150318, upper bound: 86.5145019
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -33.7523346, 74.8141327, -40.0856819, 89.0764465, -122.4187317, 114.4003525
1: -68.4929352, 111.3106232, -81.3761597, 133.1408234, -201.5089417, 192.6867828
2: -52.8990059, 107.3260803, -62.8396301, 128.0437317, -180.9427338, 170.1590424
3: -80.6281738, 128.3930664, -95.5622177, 153.6023560, -234.2193756, 223.8754883
4: -74.4618149, 122.7913513, -88.2554779, 146.8284760, -220.6127777, 210.2117615

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5153444, upper bound: 86.5144885
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5153661, upper bound: 86.5145226
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -33.7523346, 74.8141327, -39.7320976, 87.7826767, -121.2296753, 114.0753784
1: -68.4929352, 111.3106232, -80.5433502, 130.9316101, -199.4245453, 191.8539734
2: -52.8990059, 107.3260803, -62.2300110, 126.1913528, -179.0903320, 169.5560913
3: -80.6281738, 128.3930664, -94.6482086, 151.1829681, -231.8111420, 223.0203400
4: -74.4618149, 122.7913513, -87.4558029, 144.5842896, -218.6270142, 209.4904327

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5153444, upper bound: 86.5144885
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5153661, upper bound: 86.5145226
time: 1.22 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -48.3466988, 106.8648834, -41.1132126, 91.3592224, -138.5523987, 146.7595978
1: -97.1714554, 159.2430725, -83.6735687, 136.5204773, -233.1527710, 242.1217346
2: -75.3781586, 152.8537598, -64.5547028, 131.4346619, -206.1690826, 216.9148407
3: -114.8389359, 183.4018250, -98.1135635, 157.5955505, -271.3444824, 280.8078003
4: -106.4433746, 175.3153076, -90.5560837, 150.6625061, -254.6956787, 263.7509766

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5142873, upper bound: 86.5143266
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5142873, upper bound: 86.5143266
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -48.3466988, 106.8648834, -40.5770645, 89.6747971, -136.9702454, 146.2478943
1: -97.1714554, 159.2430725, -82.4455719, 133.7261353, -230.5073853, 240.9261322
2: -75.3781586, 152.8537598, -63.6483803, 128.9960632, -203.8919830, 216.0397644
3: -114.8389359, 183.4018250, -96.7555771, 154.4855347, -268.4993286, 279.5064392
4: -106.4433746, 175.3153076, -89.3559265, 147.7470245, -252.0419159, 262.6202393

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5142873, upper bound: 86.5143266
time: 1.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5142873, upper bound: 86.5143266
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -47.4663315, 104.9316788, -40.3523140, 89.7239685, -136.0520172, 144.0888824
1: -95.3373795, 156.2848511, -81.9160461, 134.1414948, -228.9341583, 237.4554901
2: -73.9950714, 149.9686127, -63.2576027, 128.9582672, -202.2993317, 212.7937775
3: -112.6960297, 179.9624634, -96.2043839, 154.7316742, -266.3316040, 275.5308838
4: -104.4971466, 172.0549622, -88.8385696, 147.9066925, -250.0042725, 258.8769836

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5141628, upper bound: 86.5137475
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5140505, upper bound: 86.5140446
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -48.0062180, 106.0532684, -39.9712486, 88.3617554, -135.3136902, 144.8491669
1: -96.4195557, 157.9739380, -81.0236053, 131.8251190, -227.8301697, 238.2571869
2: -74.8274612, 151.6258087, -62.6042175, 127.0063400, -201.3044739, 213.8125610
3: -113.9870529, 181.9369659, -95.2224808, 152.1914825, -265.2975159, 276.5537720
4: -105.6838379, 173.9095764, -87.9776917, 145.5467987, -249.0283051, 259.9335632

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143266, upper bound: 86.5142873
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143266, upper bound: 86.5142873
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -46.8842316, 103.6546402, -40.3523140, 89.7239685, -135.4852600, 142.8239746
1: -93.9370651, 154.4280548, -81.9160461, 134.1414948, -227.5409393, 235.5942078
2: -72.9789352, 147.9971161, -63.2576027, 128.9582672, -201.3115540, 210.8141937
3: -111.2140274, 177.7039490, -96.2043839, 154.7316742, -264.8617859, 273.2512512
4: -103.1699066, 169.8727417, -88.8385696, 147.9066925, -248.7721252, 256.7229614

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5141308, upper bound: 86.5137575
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5140482, upper bound: 86.5140576
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -47.4493866, 104.8295441, -39.9712486, 88.3617554, -134.7728424, 143.6407318
1: -95.0781021, 156.2013245, -81.0236053, 131.8251190, -226.4945221, 236.4794312
2: -73.8537979, 149.7345581, -62.6042175, 127.0063400, -200.3592224, 211.9147339
3: -112.5679779, 179.7712250, -95.2224808, 152.1914825, -263.8905334, 274.3675537
4: -104.4125900, 171.8248749, -87.9776917, 145.5467987, -247.8533630, 257.8786926

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5138725, upper bound: 86.5134536
time: 1.20 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5138079, upper bound: 86.5138079
time: 0.71 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.00 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5182752, upper bound: 86.5190413
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5182752, upper bound: 86.5191615
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5182752, upper bound: 86.5192508
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5182752, upper bound: 86.5195506
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5113671, upper bound: 86.5122348
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5110740, upper bound: 86.5121665
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5113671, upper bound: 86.5122348
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5110740, upper bound: 86.5121665
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5185123, upper bound: 86.5187404
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5194164, upper bound: 86.5190304
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5185123, upper bound: 86.5187555
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5194164, upper bound: 86.5191355
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5112280, upper bound: 86.5119972
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5087855, upper bound: 86.5087855
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5112280, upper bound: 86.5119972
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5087855, upper bound: 86.5087855
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5169338, upper bound: 86.5163088
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5166372, upper bound: 86.5163359
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5174089, upper bound: 86.5165843
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5165611, upper bound: 86.5163827
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5179034, upper bound: 86.5165036
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5171267, upper bound: 86.5164392
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5179034, upper bound: 86.5165036
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5171267, upper bound: 86.5164392
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5131244, upper bound: 86.5151342
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5131624, upper bound: 86.5152562
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5137950, upper bound: 86.5161757
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5137950, upper bound: 86.5162030
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5151408, upper bound: 86.5153017
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5152689, upper bound: 86.5152916
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5151408, upper bound: 86.5153016
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5151408, upper bound: 86.5152916
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5177835, upper bound: 86.5164936
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5177835, upper bound: 86.5164936
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5177835, upper bound: 86.5164936
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5177835, upper bound: 86.5164936
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5180195, upper bound: 86.5167874
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5180195, upper bound: 86.5167874
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5180195, upper bound: 86.5167874
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5177835, upper bound: 86.5167874
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5138234, upper bound: 86.5156012
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5136932, upper bound: 86.5150274
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5138234, upper bound: 86.5156012
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5136932, upper bound: 86.5150274
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5151419, upper bound: 86.5143892
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5153031, upper bound: 86.5148032
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5151419, upper bound: 86.5143892
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5153031, upper bound: 86.5148032
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5171679, upper bound: 86.5184784
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5162871, upper bound: 86.5179194
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5171679, upper bound: 86.5184927
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5162871, upper bound: 86.5179194
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5172348, upper bound: 86.5180476
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5159458, upper bound: 86.5156441
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5172348, upper bound: 86.5180476
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5159458, upper bound: 86.5156441
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5147956, upper bound: 86.5163528
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5143758, upper bound: 86.5161502
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5147956, upper bound: 86.5163528
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5143758, upper bound: 86.5161503
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5148357, upper bound: 86.5157709
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5142199, upper bound: 86.5142199
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5148357, upper bound: 86.5157709
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5142199, upper bound: 86.5142199
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5146508, upper bound: 86.5146326
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5143536, upper bound: 86.5129167
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5140549, upper bound: 86.5151959
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5150977, upper bound: 86.5160081
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5146088, upper bound: 86.5149642
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5143696, upper bound: 86.5149997
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5146358, upper bound: 86.5152448
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5145513, upper bound: 86.5155503
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5146475, upper bound: 86.5149496
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5145044, upper bound: 86.5150606
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5139792, upper bound: 86.5149572
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5150413, upper bound: 86.5157780
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5146056, upper bound: 86.5149807
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5145019, upper bound: 86.5151559
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5145976, upper bound: 86.5150405
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5145226, upper bound: 86.5153661
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5148232, upper bound: 86.5144582
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5148232, upper bound: 86.5144988
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5148283, upper bound: 86.5144583
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5148283, upper bound: 86.5145044
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5144810, upper bound: 86.5133047
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5146306, upper bound: 86.5135148
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5153348, upper bound: 86.5144944
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5153762, upper bound: 86.5145260
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5148668, upper bound: 86.5144581
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5148668, upper bound: 86.5144930
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5150318, upper bound: 86.5144593
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5150318, upper bound: 86.5145019
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5153444, upper bound: 86.5144885
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5153661, upper bound: 86.5145226
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5153444, upper bound: 86.5144885
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5153661, upper bound: 86.5145226
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5142873, upper bound: 86.5143266
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5142873, upper bound: 86.5143266
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5142873, upper bound: 86.5143266
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5142873, upper bound: 86.5143266
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5141628, upper bound: 86.5137475
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5140505, upper bound: 86.5140446
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5143266, upper bound: 86.5142873
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5143266, upper bound: 86.5142873
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5141308, upper bound: 86.5137575
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5140482, upper bound: 86.5140576
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5138725, upper bound: 86.5134536
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.00
Output dim: 0, lower bound: -86.5138079, upper bound: 86.5138079

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -28.0293713, 62.3899193, -26.4423676, 58.6073456, -86.6366959, 88.8322906
1: -58.1548958, 93.3556137, -54.7700119, 87.5604782, -145.7153625, 148.1256256
2: -44.4702950, 90.6243896, -41.9240685, 85.1240158, -129.5942993, 132.5484619
3: -67.5788345, 108.2222214, -63.6886063, 101.6700363, -169.2488708, 171.9108276
4: -62.0199471, 103.3522949, -58.5240135, 97.0233078, -159.0432281, 161.8762970

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5166926, upper bound: 86.5180960
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5166303, upper bound: 86.5172421
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -26.4629440, 58.6131134, -26.8460693, 59.4360161, -85.8989563, 85.4591675
1: -54.7460480, 87.5498047, -55.5746651, 88.7881165, -143.5341644, 143.1244659
2: -41.9325333, 85.0943298, -42.5499840, 86.3192062, -128.2517395, 127.6443176
3: -63.7135925, 101.6298599, -64.6515045, 103.0843887, -166.7979736, 166.2813568
4: -58.5628929, 96.9906998, -59.4086304, 98.3738480, -156.9367371, 156.3993073

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5166926, upper bound: 86.5186907
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5179686, upper bound: 86.5179686
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -28.0293713, 62.3899193, -26.5819016, 58.9363518, -86.9657135, 88.9718170
1: -58.1548958, 93.3556137, -55.0396767, 88.1038666, -146.2587585, 148.3952942
2: -44.4702950, 90.6243896, -42.1337509, 85.6240387, -130.0943298, 132.7581482
3: -67.5788345, 108.2222214, -64.0177078, 102.3052979, -169.8840942, 172.2398987
4: -62.0199471, 103.3522949, -58.8273888, 97.5897980, -159.6097412, 162.1796875

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5155003, upper bound: 86.5180737
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5175755, upper bound: 86.5184801
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5178318, upper bound: 86.5188382
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -26.4629440, 58.6131134, -27.0021362, 59.7928581, -86.2557983, 85.6152496
1: -54.7460480, 87.5498047, -55.8661766, 89.3706665, -144.1167145, 143.4159546
2: -41.9325333, 85.0943298, -42.7835388, 86.8583298, -128.7908630, 127.8778610
3: -63.7135925, 101.6298599, -65.0135880, 103.7606583, -167.4742432, 166.6434479
4: -58.5628929, 96.9906998, -59.7465515, 98.9856949, -157.5485840, 156.7372131

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5170057, upper bound: 86.5187231
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5114715, upper bound: 86.5136558
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -27.9599571, 62.2342377, -26.4423676, 58.6073456, -86.5673065, 88.6766052
1: -57.9532967, 93.1443405, -54.7700119, 87.5604782, -145.5137787, 147.9143372
2: -44.3364258, 90.3520126, -41.9240685, 85.1240158, -129.4604492, 132.2760773
3: -67.3871078, 107.9328918, -63.6886063, 101.6700363, -169.0571442, 171.6214905
4: -61.8555183, 103.0755768, -58.5240135, 97.0233078, -158.8788300, 161.5995331

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5183240, upper bound: 86.5181656
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5182936, upper bound: 86.5181628
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -26.6322479, 58.9970360, -26.8460693, 59.4360161, -86.0682678, 85.8430939
1: -55.0701447, 88.1715393, -55.5746651, 88.7881165, -143.8582611, 143.7462006
2: -42.1895142, 85.6739273, -42.5499840, 86.3192062, -128.5086823, 128.2239075
3: -64.1114655, 102.3518524, -64.6515045, 103.0843887, -167.1958160, 167.0033569
4: -58.9304657, 97.6482620, -59.4086304, 98.3738480, -157.3042908, 157.0568848

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5138303, upper bound: 86.5120620
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5136558, upper bound: 86.5114715
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -27.9599571, 62.2342377, -26.5819016, 58.9363518, -86.8963089, 88.8161392
1: -57.9532967, 93.1443405, -55.0396767, 88.1038666, -146.0571594, 148.1840210
2: -44.3364258, 90.3520126, -42.1337509, 85.6240387, -129.9604645, 132.4857483
3: -67.3871078, 107.9328918, -64.0177078, 102.3052979, -169.6923676, 171.9505768
4: -61.8555183, 103.0755768, -58.8273888, 97.5897980, -159.4453125, 161.9029541

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5178736, upper bound: 86.5181732
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5181307, upper bound: 86.5182414
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -26.6322479, 58.9970360, -27.0021362, 59.7928581, -86.4251099, 85.9991760
1: -55.0701447, 88.1715393, -55.8661766, 89.3706665, -144.4408112, 144.0377197
2: -42.1895142, 85.6739273, -42.7835388, 86.8583298, -129.0478058, 128.4574585
3: -64.1114655, 102.3518524, -65.0135880, 103.7606583, -167.8721161, 167.3654327
4: -58.9304657, 97.6482620, -59.7465515, 98.9856949, -157.9161377, 157.3948059

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5137264, upper bound: 86.5120620
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5084588, upper bound: 86.5084588
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -27.5375156, 61.1474648, -33.1510010, 73.1881256, -100.7256393, 94.2984619
1: -56.9791908, 91.4690857, -67.5838165, 108.7436676, -165.7228546, 159.0529022
2: -43.6301231, 88.8101807, -52.0604744, 105.4869919, -149.1171112, 140.8705902
3: -66.3164062, 105.9990158, -79.2040253, 125.8692093, -192.1856079, 185.2030334
4: -60.9232025, 101.2326431, -73.0597153, 120.4650650, -181.3882751, 174.2923584

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169338, upper bound: 86.5163088
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169338, upper bound: 86.5163088
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -27.1865826, 60.3173256, -33.0883102, 73.0464783, -100.2330627, 93.4056320
1: -56.2317429, 90.2363739, -67.4484482, 108.5329971, -164.7647247, 157.6848145
2: -43.0647316, 87.6187973, -51.9585571, 105.2807236, -148.3454590, 139.5773621
3: -65.4428406, 104.5808792, -79.0493698, 125.6248169, -191.0676575, 183.6302338
4: -60.1607895, 99.8657227, -72.9225235, 120.2288818, -180.3896790, 172.7882233

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5166372, upper bound: 86.5163359
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5166372, upper bound: 86.5163359
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -27.4632435, 61.2083778, -35.4943504, 78.6009979, -106.0642395, 96.7027054
1: -56.8863754, 91.5717239, -72.6358032, 116.8737335, -173.7600861, 164.2075195
2: -43.5362549, 88.7687607, -55.8597946, 113.3109360, -156.8471832, 144.6285553
3: -66.1634750, 106.1113510, -84.9570847, 135.3140106, -201.4774780, 191.0684357
4: -60.7741165, 101.2978363, -78.2670059, 129.4452057, -190.2193298, 179.5648346

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5174089, upper bound: 86.5165843
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5174089, upper bound: 86.5165843
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -26.6849346, 59.4923859, -34.9683723, 77.5260544, -104.2109756, 94.4607391
1: -55.1317329, 88.9918594, -71.5062637, 115.2537003, -170.3854370, 160.4981232
2: -42.2273827, 86.2521362, -55.0031242, 111.6871338, -153.9145203, 141.2552338
3: -64.2359619, 103.0902100, -83.6710358, 133.4108276, -197.6467743, 186.7611847
4: -59.0162659, 98.4377670, -77.1001968, 127.6296692, -186.6459351, 175.5379639

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5165611, upper bound: 86.5163827
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5165611, upper bound: 86.5163827
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -25.9809933, 57.6246948, -37.0079689, 82.3425980, -108.3235779, 94.6326599
1: -53.6692924, 86.0546112, -75.8468933, 122.6914291, -176.3607178, 161.9015045
2: -41.1399498, 83.5629272, -58.2833328, 118.7524643, -159.8924103, 141.8462372
3: -62.5074043, 99.8980637, -88.6719666, 141.9727783, -204.4801788, 188.5700378
4: -57.5015564, 95.2752075, -81.5984879, 135.7218323, -193.2233734, 176.8736877

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5155756, upper bound: 86.5158587
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5155571, upper bound: 86.5157853
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -25.5595093, 56.6809845, -36.4671021, 81.2300262, -106.7895355, 93.1480865
1: -52.6779709, 84.6682053, -74.6835861, 121.0215302, -173.6994934, 159.3517914
2: -40.4051285, 82.1968842, -57.4006805, 117.0793839, -157.4844818, 139.5975647
3: -61.4477158, 98.2723236, -87.3498154, 140.0116425, -201.4593353, 185.6221008
4: -56.5318146, 93.7075272, -80.3985596, 133.8437805, -190.3755951, 174.1060638

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5162829, upper bound: 86.5162449
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5162649, upper bound: 86.5161707
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -25.9809933, 57.6246948, -35.7812653, 79.2719421, -105.2529144, 93.4059601
1: -53.6692924, 86.0546112, -73.2410202, 117.9032593, -171.5725555, 159.2956238
2: -41.1399498, 83.5629272, -56.3148956, 114.3314972, -155.4714508, 139.8778076
3: -62.5074043, 99.8980637, -85.6648254, 136.5206299, -199.0280151, 185.5628967
4: -57.5015564, 95.2752075, -78.8962021, 130.5774994, -188.0790405, 174.1714172

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5155063, upper bound: 86.5158343
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5154407, upper bound: 86.5157173
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -25.5595093, 56.6809845, -35.2481575, 78.1815948, -103.7410965, 91.9291382
1: -52.6779709, 84.6682053, -72.0994339, 116.2605667, -168.9385071, 156.7676392
2: -40.4051285, 82.1968842, -55.4472733, 112.6868134, -153.0919189, 137.6441650
3: -61.4477158, 98.2723236, -84.3640442, 134.5860291, -196.0337219, 182.6363525
4: -56.5318146, 93.7075272, -77.7147064, 128.7378998, -185.2697144, 171.4222412

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5162070, upper bound: 86.5162220
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5161460, upper bound: 86.5160973
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -27.5375156, 61.1474648, -33.4920731, 73.9373932, -101.4749069, 94.6395416
1: -56.9791908, 91.4690857, -68.2814407, 109.9154358, -166.8946228, 159.7505188
2: -43.6301231, 88.8101807, -52.6077919, 106.6270599, -150.2571869, 141.4179688
3: -66.3164062, 105.9990158, -80.0148697, 127.2453995, -193.5617981, 186.0138855
4: -60.9232025, 101.2326431, -73.8056107, 121.7455368, -182.6687317, 175.0382385

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5131244, upper bound: 86.5151342
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5131244, upper bound: 86.5151342
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -27.1865826, 60.3173256, -33.4290543, 73.7947845, -100.9813690, 93.7463837
1: -56.2317429, 90.2363739, -68.1455078, 109.7033005, -165.9350128, 158.3818817
2: -43.0647316, 87.6187973, -52.5053711, 106.4195328, -149.4842682, 140.1241760
3: -65.4428406, 104.5808792, -79.8595200, 126.9990616, -192.4418945, 184.4403534
4: -60.1607895, 99.8657227, -73.6676407, 121.5077209, -181.6685181, 173.5333405

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5131624, upper bound: 86.5152562
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5131624, upper bound: 86.5152562
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -25.5777454, 56.8153038, -36.0564423, 79.8585358, -105.4362793, 92.8717499
1: -52.7732658, 84.9462051, -73.8144073, 118.8341675, -171.6074371, 158.7605896
2: -40.4605103, 82.4564743, -56.7529182, 115.2225189, -155.6830292, 139.2093658
3: -61.5112076, 98.3912964, -86.3179169, 137.6053925, -199.1165619, 184.7091370
4: -56.5612106, 94.0284576, -79.5011902, 131.5942993, -188.1555023, 173.5296478

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5137950, upper bound: 86.5161757
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5137950, upper bound: 86.5161757
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -28.4643917, 63.3943367, -36.0564423, 79.8585358, -108.3229141, 99.4507751
1: -59.0223236, 94.8573227, -73.8144073, 118.8341675, -177.8564911, 168.6717224
2: -45.1443787, 92.0266190, -56.7529182, 115.2225189, -160.3668976, 148.7795105
3: -68.6034927, 109.9283752, -86.3179169, 137.6053925, -206.2088776, 196.2462921
4: -62.9724083, 104.9814987, -79.5011902, 131.5942993, -194.5666809, 184.4826508

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5137950, upper bound: 86.5162030
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5137950, upper bound: 86.5162030
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -25.7216759, 57.1807518, -37.1226807, 82.6482925, -108.3699646, 94.3034363
1: -53.2611198, 85.3803482, -76.0933380, 123.2080002, -176.4691010, 161.4736633
2: -40.7698097, 82.8874512, -58.4607544, 119.2538605, -160.0236664, 141.3482056
3: -61.9394913, 99.0798950, -88.9524155, 142.5615082, -204.5010071, 188.0323181
4: -56.9099884, 94.5825729, -81.8390350, 136.2774048, -193.1873932, 176.4216003

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5173196, upper bound: 86.5173324
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5167220, upper bound: 86.5171824
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5167375, upper bound: 86.5168747
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -27.3802567, 61.3843155, -37.0402412, 82.4403687, -109.8206253, 98.4245605
1: -56.8446274, 91.8698502, -75.8947754, 122.8831024, -179.7277222, 167.7646179
2: -43.4572144, 89.0758209, -58.3248405, 118.9364014, -162.3935852, 147.4006653
3: -66.0374146, 106.7157669, -88.7402115, 142.1879425, -208.2253418, 195.4559784
4: -60.6091080, 101.7655106, -81.6596756, 135.9204559, -196.5295715, 183.4251862

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5167037, upper bound: 86.5169867
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5167037, upper bound: 86.5166970
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -25.7216759, 57.1807518, -35.9009476, 79.5857239, -105.3074036, 93.0816956
1: -53.2611198, 85.3803482, -73.5043335, 118.4299927, -171.6911011, 158.8846741
2: -40.7698097, 82.8874512, -56.5080986, 114.8274994, -155.5972748, 139.3955536
3: -61.9394913, 99.0798950, -85.9579926, 137.1418152, -199.0812988, 185.0378571
4: -56.9099884, 94.5825729, -79.1531906, 131.1363678, -188.0463257, 173.7357635

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5146187, upper bound: 86.5150191
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5144519, upper bound: 86.5149131
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5135035, upper bound: 86.5142881
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -27.3802567, 61.3843155, -35.7951851, 79.3321609, -106.7124176, 97.1794968
1: -56.8446274, 91.8698502, -73.2494888, 118.0454941, -174.8900909, 165.1193390
2: -43.4572144, 89.0758209, -56.3312798, 114.4448242, -157.9019775, 145.4070892
3: -66.0374146, 106.7157669, -85.6853180, 136.6980896, -202.7354889, 192.4010468
4: -60.6091080, 101.7655106, -78.9209061, 130.7052002, -191.3143005, 180.6864166

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5144688, upper bound: 86.5148998
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5135464, upper bound: 86.5142778
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -27.0612755, 60.4953613, -37.0138931, 82.3858871, -109.4471588, 97.5092545
1: -56.1623917, 90.5420151, -75.9018860, 122.7563477, -178.9187012, 166.4439087
2: -42.9302521, 87.7347107, -58.3063850, 118.8230972, -161.7533264, 146.0410919
3: -65.2457962, 104.9088593, -88.7042770, 142.0440674, -207.2898560, 193.6131287
4: -59.8645325, 100.1728973, -81.6085892, 135.8098145, -195.6743164, 181.7814484

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148345, upper bound: 86.5136454
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149451, upper bound: 86.5136160
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -25.8590145, 57.5004768, -37.0138931, 82.3858871, -108.2448959, 94.5143738
1: -53.5140343, 85.9069519, -75.9018860, 122.7563477, -176.2703552, 161.8088226
2: -40.9763374, 83.3832779, -58.3063850, 118.8230972, -159.7994385, 141.6896667
3: -62.2555428, 99.6786499, -88.7042770, 142.0440674, -204.2995911, 188.3829346
4: -57.2089386, 95.1379852, -81.6085892, 135.8098145, -193.0187531, 176.7465668

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148345, upper bound: 86.5136454
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148345, upper bound: 86.5136160
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -27.0612755, 60.4953613, -37.1226807, 82.6482925, -109.7095642, 97.6180420
1: -56.1623917, 90.5420151, -76.0933380, 123.2080002, -179.3703766, 166.6353455
2: -42.9302521, 87.7347107, -58.4607544, 119.2538605, -162.1841125, 146.1954651
3: -65.2457962, 104.9088593, -88.9524155, 142.5615082, -207.8073120, 193.8612671
4: -59.8645325, 100.1728973, -81.8390350, 136.2774048, -196.1419220, 182.0119019

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143523, upper bound: 86.5136454
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143523, upper bound: 86.5135270
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -25.8590145, 57.5004768, -37.1226807, 82.6482925, -108.5073013, 94.6231537
1: -53.5140343, 85.9069519, -76.0933380, 123.2080002, -176.7220306, 162.0002594
2: -40.9763374, 83.3832779, -58.4607544, 119.2538605, -160.2301941, 141.8440247
3: -62.2555428, 99.6786499, -88.9524155, 142.5615082, -204.8170166, 188.6310730
4: -57.2089386, 95.1379852, -81.8390350, 136.2774048, -193.4863434, 176.9770050

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143523, upper bound: 86.5136454
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143523, upper bound: 86.5135270
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -29.1783810, 65.6018753, -36.9197769, 82.1623459, -111.3407288, 102.5216522
1: -60.6789742, 98.3641815, -75.6833191, 122.4114914, -183.0904541, 174.0475006
2: -46.3428307, 95.2231369, -58.1524544, 118.4932327, -164.8360443, 153.3755951
3: -70.4400864, 114.0466614, -88.4665527, 141.6558685, -212.0959473, 202.5132141
4: -64.5630951, 108.8419495, -81.4035568, 135.4365692, -199.9996490, 190.2454834

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148345, upper bound: 86.5153938
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5156765, upper bound: 86.5154381
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -27.8189068, 62.3239594, -36.9197769, 82.1623459, -109.9812546, 99.2437363
1: -57.7220688, 93.3147507, -75.6833191, 122.4114914, -180.1335297, 168.9980774
2: -44.1496162, 90.4649658, -58.1524544, 118.4932327, -162.6428223, 148.6174164
3: -67.0813217, 108.3722610, -88.4665527, 141.6558685, -208.7371826, 196.8388062
4: -61.5620613, 103.3378372, -81.4035568, 135.4365692, -196.9986267, 184.7413940

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148345, upper bound: 86.5153938
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149451, upper bound: 86.5154381
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -29.1783810, 65.6018753, -37.0402412, 82.4403687, -111.6187515, 102.6421204
1: -60.6789742, 98.3641815, -75.8947754, 122.8831024, -183.5620728, 174.2589569
2: -46.3428307, 95.2231369, -58.3248405, 118.9364014, -165.2792358, 153.5479584
3: -70.4400864, 114.0466614, -88.7402115, 142.1879425, -212.6280212, 202.7868652
4: -64.5630951, 108.8419495, -81.6596756, 135.9204559, -200.4835510, 190.5016174

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143523, upper bound: 86.5153938
time: 1.28 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143523, upper bound: 86.5154381
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -27.8189068, 62.3239594, -37.0402412, 82.4403687, -110.2592773, 99.3641968
1: -57.7220688, 93.3147507, -75.8947754, 122.8831024, -180.6051636, 169.2094879
2: -44.1496162, 90.4649658, -58.3248405, 118.9364014, -163.0860138, 148.7898102
3: -67.0813217, 108.3722610, -88.7402115, 142.1879425, -209.2692566, 197.1124725
4: -61.5620613, 103.3378372, -81.6596756, 135.9204559, -197.4825134, 184.9975128

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5155840, upper bound: 86.5153938
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5156379, upper bound: 86.5154381
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -25.7428017, 57.1371803, -35.4241982, 78.2606354, -104.0034332, 92.5613785
1: -53.0524254, 85.4214020, -72.4177856, 116.3816147, -169.4340210, 157.8391876
2: -40.7159195, 82.8728790, -55.7198029, 112.9318314, -153.6477051, 138.5926819
3: -61.8795166, 98.9003296, -84.7600861, 134.7791748, -196.6586914, 183.6604156
4: -56.9145889, 94.5157013, -78.0998230, 128.8993073, -185.8138733, 172.6155243

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5162896, upper bound: 86.5159626
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5163301, upper bound: 86.5156004
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -28.3604126, 63.1573906, -36.1378021, 80.0470886, -108.4075012, 99.2951965
1: -58.7424965, 94.5265808, -74.0030823, 119.0674973, -177.8099976, 168.5296631
2: -44.9527283, 91.6393509, -56.8876266, 115.4802551, -160.4329834, 148.5269623
3: -68.3249741, 109.5111542, -86.5337219, 137.8710938, -206.1960602, 196.0448761
4: -62.7322502, 104.5666809, -79.6820450, 131.8813934, -194.6136475, 184.2487183

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5170738, upper bound: 86.5155289
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5170738, upper bound: 86.5155289
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -25.7428017, 57.1371803, -35.5465240, 78.5542297, -104.2970276, 92.6837006
1: -53.0524254, 85.4214020, -72.6468964, 116.8895187, -169.9419403, 158.0682526
2: -40.7159195, 82.8728790, -55.9050179, 113.3906174, -154.1065216, 138.7778931
3: -61.8795166, 98.9003296, -85.0413361, 135.3636780, -197.2431793, 183.9416656
4: -56.9145889, 94.5157013, -78.3658905, 129.4265137, -186.3410645, 172.8815918

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5126721, upper bound: 86.5151887
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5131905, upper bound: 86.5149206
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5136932, upper bound: 86.5150274
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5136932, upper bound: 86.5150274
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -28.3604126, 63.1573906, -36.2581635, 80.3262939, -108.6867065, 99.4155502
1: -58.7424965, 94.5265808, -74.2217102, 119.5489655, -178.2914581, 168.7482910
2: -44.9527283, 91.6393509, -57.0672417, 115.9224319, -160.8751526, 148.7065887
3: -68.3249741, 109.5111542, -86.8075562, 138.4415894, -206.7665710, 196.3187103
4: -62.7322502, 104.5666809, -79.9424362, 132.3672638, -195.0995178, 184.5091248

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.39 + 416.90 = 420.28 seconds
