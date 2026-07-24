## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 807.3886655422


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953)
1: (-373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194)
2: (-542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741)
3: (-209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672)
4: (-604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.77 + 1.77 = 3.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -809.0066789, upper bound: 809.0066789

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4691278, upper bound: 808.5062695
time: 0.72 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0038423, upper bound: 809.0038429
time: 0.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.55 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 4, lower bound: -806.4691278, upper bound: 808.5062695
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 4, lower bound: -809.0038423, upper bound: 809.0038429

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -355.7917175, 298.0696411, -461.5992737, 385.9967346, -741.7883301, 759.6687012
1: -284.4114380, 290.6557922, -370.1784058, 374.3149719, -658.7263794, 660.8341675
2: -411.3229675, 317.9904480, -538.0414429, 408.4810486, -819.8040161, 856.0318604
3: -162.9068451, 405.6134033, -208.3363800, 526.5999756, -689.5068359, 613.9497070
4: -458.9186401, 313.6753845, -599.3458252, 404.4221802, -863.3408203, 913.0212402

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.4684386, upper bound: 806.4684386
time: 0.67 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4684386, upper bound: 808.5062695
time: 0.70 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -455.0569763, 380.6590271, -465.2375793, 388.7125854, -843.7695312, 845.8966064
1: -364.9006348, 369.0178833, -373.0900269, 376.9524841, -741.8531494, 742.1078491
2: -530.4962158, 402.6548462, -542.3095093, 411.3765564, -941.8726807, 944.9643555
3: -205.3341675, 519.1251221, -209.7973175, 530.6040649, -735.9381104, 728.9224243
4: -590.9782715, 398.9001770, -604.0916138, 407.2537537, -998.2319336, 1002.9916992

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5062695, upper bound: 806.4691278
time: 0.67 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5062695, upper bound: 806.4691278
time: 0.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.99 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.99
Output dim: 4, lower bound: -806.4684386, upper bound: 806.4684386
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 4, lower bound: -806.4684386, upper bound: 808.5062695
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 4, lower bound: -808.5062695, upper bound: 806.4691278
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 4, lower bound: -808.5062695, upper bound: 806.4691278

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -355.7917175, 298.0696411, -455.0569763, 380.6590271, -736.4506836, 753.1265869
1: -284.4114380, 290.6557922, -364.9006348, 369.0178833, -653.4293213, 655.5563965
2: -411.3229675, 317.9904480, -530.4962158, 402.6548462, -813.9777832, 848.4865723
3: -162.9068451, 405.6134033, -205.3341675, 519.1251221, -682.0319824, 610.9474487
4: -458.9186401, 313.6753845, -590.9782715, 398.9001770, -857.8187256, 904.6536865

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.4198038, upper bound: 806.4683075
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4683305, upper bound: 808.3347836
time: 0.69 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -455.0569763, 380.6590271, -355.7917175, 298.0696411, -753.1265869, 736.4506836
1: -364.9006348, 369.0178833, -284.4114380, 290.6557922, -655.5563965, 653.4293213
2: -530.4962158, 402.6548462, -411.3229675, 317.9904480, -848.4865723, 813.9777832
3: -205.3341675, 519.1251221, -162.9068451, 405.6134033, -610.9473877, 682.0319824
4: -590.9782715, 398.9001770, -458.9186401, 313.6753845, -904.6536865, 857.8186646

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5061989, upper bound: 806.4689921
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3347828, upper bound: 806.4689246
time: 0.54 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -455.0569763, 380.6590271, -455.0569763, 380.6590271, -835.7160034, 835.7160034
1: -364.9006348, 369.0178833, -364.9006348, 369.0178833, -733.9185181, 733.9185181
2: -530.4962158, 402.6548462, -530.4962158, 402.6548462, -933.1510620, 933.1510620
3: -205.3341675, 519.1251221, -205.3341675, 519.1251221, -724.4591675, 724.4591675
4: -590.9782715, 398.9001770, -590.9782715, 398.9001770, -989.8784180, 989.8782959

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5062003, upper bound: 808.5813317
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3347836, upper bound: 808.5812559
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.11 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.11
Output dim: 4, lower bound: -806.4198038, upper bound: 806.4683075
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 4, lower bound: -806.4683305, upper bound: 808.3347836
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 4, lower bound: -808.5061989, upper bound: 806.4689921
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 4, lower bound: -808.3347828, upper bound: 806.4689246
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 4, lower bound: -808.5062003, upper bound: 808.5813317
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 4, lower bound: -808.3347836, upper bound: 808.5812559

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -346.6198730, 290.6638184, -454.6670532, 380.3390503, -726.9589233, 745.3308716
1: -276.9523926, 283.5098572, -364.5827637, 368.7083130, -645.6607056, 648.0926514
2: -400.2306824, 310.2190247, -530.0238647, 402.3171387, -802.5477905, 840.2429199
3: -158.6000671, 395.0196533, -205.1652374, 518.6757812, -677.2758789, 600.1846924
4: -446.5895996, 306.0231628, -590.4550171, 398.5679626, -845.1575928, 896.4781494

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4689246, upper bound: 808.3347836
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4689246, upper bound: 808.3347828
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -437.7723999, 367.4412842, -354.9750061, 297.3641357, -735.1364136, 722.4162598
1: -350.6974182, 356.2439270, -283.7489929, 289.9807739, -640.6782227, 639.9929199
2: -509.7792053, 388.8671265, -410.3548889, 317.2487183, -827.0278320, 799.2220459
3: -198.0632477, 499.4380493, -162.5610352, 404.6816101, -602.7448730, 661.9990845
4: -568.3402710, 385.4368286, -457.8487854, 312.9402161, -881.2805176, 843.2856445

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4603622, upper bound: 806.4634676
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0122074, upper bound: 806.4204047
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0122074, upper bound: 806.4689246
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -444.8894653, 372.2908936, -355.4672241, 297.8057861, -742.6952515, 727.7581177
1: -356.6080017, 360.9292603, -284.1471252, 290.4016113, -647.0095825, 645.0764160
2: -518.1353760, 393.8309326, -410.9306946, 317.7142334, -835.8496094, 804.7615967
3: -200.9437408, 507.3840332, -162.7540436, 405.2380676, -606.1817017, 670.1379395
4: -577.2812500, 390.2192383, -458.4822998, 313.4028625, -890.6840820, 848.7015381

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3342111, upper bound: 806.4575129
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0122074, upper bound: 806.4204047
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0122074, upper bound: 806.4689246
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -437.7723999, 367.4412842, -453.9830627, 379.6953125, -817.4675293, 821.4243164
1: -350.6974182, 356.2439270, -364.0279846, 368.1007385, -718.7981567, 720.2719116
2: -509.7792053, 388.8671265, -529.2156982, 401.6600037, -911.4390869, 918.0828247
3: -198.0632477, 499.4380493, -204.8486328, 517.8768311, -715.9400635, 704.2866821
4: -568.3402710, 385.4368286, -589.5609741, 397.9028931, -966.2431641, 974.9977417

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5811934
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5811934
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -444.8894653, 372.2908936, -454.6670532, 380.3390503, -825.2284546, 826.9579468
1: -356.6080017, 360.9292603, -364.5827637, 368.7083130, -725.3162842, 725.5120239
2: -518.1353760, 393.8309326, -530.0238647, 402.3171387, -920.4525146, 923.8547974
3: -200.9437408, 507.3840332, -205.1652374, 518.6757812, -719.6195068, 712.5491333
4: -577.2812500, 390.2192383, -590.4550171, 398.5679626, -975.8492432, 980.6741943

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5812554
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5812554
time: 0.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.20 seconds
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 4, lower bound: -806.4689246, upper bound: 808.3347836
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 4, lower bound: -806.4689246, upper bound: 808.3347828
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 4, lower bound: -808.0122074, upper bound: 806.4204047
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 4, lower bound: -808.0122074, upper bound: 806.4689246
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 4, lower bound: -808.0122074, upper bound: 806.4204047
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 4, lower bound: -808.0122074, upper bound: 806.4689246
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5811934
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5811934
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5812554
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5812554

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -346.6198730, 290.6638184, -437.7723999, 367.4412842, -714.0611572, 728.4360962
1: -276.9523926, 283.5098572, -350.6974182, 356.2439270, -633.1962891, 634.2072754
2: -400.2306824, 310.2190247, -509.7792053, 388.8671265, -789.0977173, 819.9982300
3: -158.6000671, 395.0196533, -198.0632477, 499.4380493, -658.0380249, 593.0827637
4: -446.5895996, 306.0231628, -568.3402710, 385.4368286, -832.0264282, 874.3634033

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8379420, upper bound: 807.9472276
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2181072, upper bound: 808.1885050
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -346.6198730, 290.6638184, -444.8894653, 372.2908936, -718.9107666, 735.5532837
1: -276.9523926, 283.5098572, -356.6080017, 360.9292603, -637.8815918, 640.1177979
2: -400.2306824, 310.2190247, -518.1353760, 393.8309326, -794.0615845, 828.3543701
3: -158.6000671, 395.0196533, -200.9437408, 507.3840332, -665.9840698, 595.9632568
4: -446.5895996, 306.0231628, -577.2812500, 390.2192383, -836.8088379, 883.3044434

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8379420, upper bound: 807.9988370
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2181072, upper bound: 808.1885050
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -437.7723999, 367.4412842, -341.3818359, 288.1004028, -725.8728027, 708.8230591
1: -350.6974182, 356.2439270, -272.6688843, 281.0004883, -631.6978760, 628.9125977
2: -509.7792053, 388.8671265, -394.2149963, 307.6066589, -817.3858643, 783.0821533
3: -198.0632477, 499.4380493, -157.2244263, 389.4662170, -587.5294800, 656.6624756
4: -568.3402710, 385.4368286, -440.2426147, 303.5801392, -871.9202881, 825.6794434

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.1286589, upper bound: 806.4180055
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4874518, upper bound: 806.4198650
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -437.7723999, 367.4412842, -346.6198730, 290.6638184, -728.4360352, 714.0611572
1: -350.6974182, 356.2439270, -276.9523926, 283.5098572, -634.2072144, 633.1962280
2: -509.7792053, 388.8671265, -400.2306824, 310.2190247, -819.9982300, 789.0977783
3: -198.0632477, 499.4380493, -158.6000671, 395.0196533, -593.0827637, 658.0380249
4: -568.3402710, 385.4368286, -446.5895996, 306.0231628, -874.3634033, 832.0264282

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.1286589, upper bound: 806.4251288
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4874518, upper bound: 806.4689680
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -444.8894653, 372.2908936, -341.3818359, 288.1004028, -732.9898682, 713.6727295
1: -356.6080017, 360.9292603, -272.6688843, 281.0004883, -637.6085205, 633.5980835
2: -518.1353760, 393.8309326, -394.2149963, 307.6066589, -825.7420654, 788.0458984
3: -200.9437408, 507.3840332, -157.2244263, 389.4662170, -590.4099731, 664.6084595
4: -577.2812500, 390.2192383, -440.2426147, 303.5801392, -880.8613281, 830.4617920

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.2180048, upper bound: 806.4200040
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0122074, upper bound: 806.4202723
time: 3.95 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -444.8894653, 372.2908936, -346.6198730, 290.6638184, -735.5532227, 718.9107056
1: -356.6080017, 360.9292603, -276.9523926, 283.5098572, -640.1178589, 637.8816528
2: -518.1353760, 393.8309326, -400.2306824, 310.2190247, -828.3543701, 794.0615845
3: -200.9437408, 507.3840332, -158.6000671, 395.0196533, -595.9631958, 665.9840698
4: -577.2812500, 390.2192383, -446.5895996, 306.0231628, -883.3044434, 836.8088379

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.2180048, upper bound: 806.4686430
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0122074, upper bound: 806.4687246
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -437.7723999, 367.4412842, -437.7723999, 367.4412842, -805.2136230, 805.2136230
1: -350.6974182, 356.2439270, -350.6974182, 356.2439270, -706.9413452, 706.9413452
2: -509.7792053, 388.8671265, -509.7792053, 388.8671265, -898.6461792, 898.6463013
3: -198.0632477, 499.4380493, -198.0632477, 499.4380493, -697.5012817, 697.5012817
4: -568.3402710, 385.4368286, -568.3402710, 385.4368286, -953.7770996, 953.7770996

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8217030, upper bound: 808.4947826
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0028057, upper bound: 808.5811780
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -437.7723999, 367.4412842, -444.8894653, 372.2908936, -810.0632935, 812.3307495
1: -350.6974182, 356.2439270, -356.6080017, 360.9292603, -711.6267090, 712.8518677
2: -509.7792053, 388.8671265, -518.1353760, 393.8309326, -903.6101074, 907.0024414
3: -198.0632477, 499.4380493, -200.9437408, 507.3840332, -705.4472656, 700.3817139
4: -568.3402710, 385.4368286, -577.2812500, 390.2192383, -958.5594482, 962.7180786

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8217030, upper bound: 808.4947861
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0028057, upper bound: 808.5811780
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -444.8894653, 372.2908936, -437.7723999, 367.4412842, -812.3307495, 810.0632324
1: -356.6080017, 360.9292603, -350.6974182, 356.2439270, -712.8519287, 711.6267090
2: -518.1353760, 393.8309326, -509.7792053, 388.8671265, -907.0024414, 903.6101074
3: -200.9437408, 507.3840332, -198.0632477, 499.4380493, -700.3817749, 705.4472656
4: -577.2812500, 390.2192383, -568.3402710, 385.4368286, -962.7180786, 958.5593872

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2532813, upper bound: 808.2641135
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5811857, upper bound: 808.5812322
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -444.8894653, 372.2908936, -444.8894653, 372.2908936, -817.1803589, 817.1803589
1: -356.6080017, 360.9292603, -356.6080017, 360.9292603, -717.5372314, 717.5372314
2: -518.1353760, 393.8309326, -518.1353760, 393.8309326, -911.9663086, 911.9663086
3: -200.9437408, 507.3840332, -200.9437408, 507.3840332, -708.3276978, 708.3277588
4: -577.2812500, 390.2192383, -577.2812500, 390.2192383, -967.5004883, 967.5004883

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2532813, upper bound: 808.2641135
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5811857, upper bound: 808.5812322
time: 0.83 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.37 seconds
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 4, lower bound: -805.8379420, upper bound: 807.9472276
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 4, lower bound: -806.2181072, upper bound: 808.1885050
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 4, lower bound: -805.8379420, upper bound: 807.9988370
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 4, lower bound: -806.2181072, upper bound: 808.1885050
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.37
Output dim: 4, lower bound: -805.1286589, upper bound: 806.4180055
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 4, lower bound: -808.4874518, upper bound: 806.4198650
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.37
Output dim: 4, lower bound: -805.1286589, upper bound: 806.4251288
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 4, lower bound: -808.4874518, upper bound: 806.4689680
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.37
Output dim: 4, lower bound: -807.2180048, upper bound: 806.4200040
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 4, lower bound: -808.0122074, upper bound: 806.4202723
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.37
Output dim: 4, lower bound: -807.2180048, upper bound: 806.4686430
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 4, lower bound: -808.0122074, upper bound: 806.4687246
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 4, lower bound: -808.8217030, upper bound: 808.4947826
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 4, lower bound: -809.0028057, upper bound: 808.5811780
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 4, lower bound: -808.8217030, upper bound: 808.4947861
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 4, lower bound: -809.0028057, upper bound: 808.5811780
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 4, lower bound: -808.2532813, upper bound: 808.2641135
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 4, lower bound: -808.5811857, upper bound: 808.5812322
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 4, lower bound: -808.2532813, upper bound: 808.2641135
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 4, lower bound: -808.5811857, upper bound: 808.5812322

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -243.7677917, 213.4292908, -421.7364197, 355.5565491, -599.3243408, 635.1656494
1: -193.7469482, 208.7287445, -337.7390747, 344.6294861, -538.3763428, 546.4677734
2: -278.9052429, 229.1844482, -490.8784485, 376.2731018, -655.1781006, 720.0628662
3: -116.8389282, 279.3439636, -191.5167999, 481.3723755, -598.2111816, 470.8607483
4: -312.4699707, 226.3032074, -547.4237671, 373.1480103, -685.6177979, 773.7269287

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.5935406, upper bound: 807.2592837
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5942979, upper bound: 807.8635989
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -339.1815186, 285.2177429, -437.7723999, 367.4412842, -706.6228027, 722.9901123
1: -270.9281616, 278.1785583, -350.6974182, 356.2439270, -627.1721191, 628.8759766
2: -391.5458069, 304.4880066, -509.7792053, 388.8671265, -780.4128418, 814.2671509
3: -155.5240631, 386.5900574, -198.0632477, 499.4380493, -654.9620972, 584.6531372
4: -436.9592896, 300.3887939, -568.3402710, 385.4368286, -822.3961182, 868.7290649

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7842918, upper bound: 807.5131779
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7846018, upper bound: 808.2353429
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -243.7677917, 213.4292908, -428.8754883, 360.4992065, -604.2669067, 642.3046875
1: -193.7469482, 208.7287445, -343.6613159, 349.3997192, -543.1466064, 552.3900757
2: -278.9052429, 229.1844482, -499.2624817, 381.3300781, -660.2352295, 728.4468994
3: -116.8389282, 279.3439636, -194.4609222, 489.3500671, -606.1889648, 473.8048706
4: -312.4699707, 226.3032074, -556.4102783, 378.0300903, -690.5000610, 782.7134399

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8378195, upper bound: 807.9988028
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8378312, upper bound: 807.9959642
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -339.1815186, 285.2177429, -444.8894653, 372.2908936, -711.4724121, 730.1071777
1: -270.9281616, 278.1785583, -356.6080017, 360.9292603, -631.8574219, 634.7865601
2: -391.5458069, 304.4880066, -518.1353760, 393.8309326, -785.3767090, 822.6232910
3: -155.5240631, 386.5900574, -200.9437408, 507.3840332, -662.9080811, 587.5335693
4: -436.9592896, 300.3887939, -577.2812500, 390.2192383, -827.1785278, 877.6700439

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.1253426, upper bound: 808.1825999
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2179647, upper bound: 808.1885050
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -434.0058594, 364.2015076, -341.3818359, 288.1004028, -722.1062622, 705.5832520
1: -347.6188049, 353.1040955, -272.6688843, 281.0004883, -628.6192017, 625.7727051
2: -505.2263184, 385.4602966, -394.2149963, 307.6066589, -812.8330078, 779.6752930
3: -196.3519897, 495.0065613, -157.2244263, 389.4662170, -585.8182373, 652.2309570
4: -563.3115845, 382.0725098, -440.2426147, 303.5801392, -866.8915405, 822.3151245

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.1904402, upper bound: 804.9128426
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4309983, upper bound: 806.2183171
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -434.0058594, 364.2015076, -346.6198730, 290.6638184, -724.6695557, 710.8212280
1: -347.6188049, 353.1040955, -276.9523926, 283.5098572, -631.1284790, 630.0563965
2: -505.2263184, 385.4602966, -400.2306824, 310.2190247, -815.4453125, 785.6909180
3: -196.3519897, 495.0065613, -158.6000671, 395.0196533, -591.3715210, 653.6065063
4: -563.3115845, 382.0725098, -446.5895996, 306.0231628, -869.3347168, 828.6621094

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9383225, upper bound: 805.8379355
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4182578, upper bound: 806.2183171
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -436.6898499, 365.1800842, -341.3818359, 288.1004028, -724.7902832, 706.5618286
1: -349.9454956, 354.0716248, -272.6688843, 281.0004883, -630.9459839, 626.7404175
2: -508.4296265, 386.5229187, -394.2149963, 307.6066589, -816.0362549, 780.7379150
3: -197.2324371, 497.9155884, -157.2244263, 389.4662170, -586.6986694, 655.1400146
4: -566.5614014, 382.8739319, -440.2426147, 303.5801392, -870.1414795, 823.1163940

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7042999, upper bound: 804.9124045
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8261256, upper bound: 806.2179647
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -436.6898499, 365.1800842, -346.6198730, 290.6638184, -727.3536377, 711.7998657
1: -349.9454956, 354.0716248, -276.9523926, 283.5098572, -633.4552612, 631.0240479
2: -508.4296265, 386.5229187, -400.2306824, 310.2190247, -818.6486816, 786.7535400
3: -197.2324371, 497.9155884, -158.6000671, 395.0196533, -592.2518921, 656.5156250
4: -566.5614014, 382.8739319, -446.5895996, 306.0231628, -872.5845947, 829.4634399

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.1529059, upper bound: 805.8371398
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8261256, upper bound: 806.2179647
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -394.8950500, 334.1777039, -437.7723999, 367.4412842, -762.3363037, 771.9500732
1: -316.1249084, 323.8029480, -350.6974182, 356.2439270, -672.3688354, 674.5003662
2: -459.4748840, 353.5050354, -509.7792053, 388.8671265, -848.3419800, 863.2839966
3: -179.7761841, 450.4103699, -198.0632477, 499.4380493, -679.2141113, 648.4735718
4: -512.8085938, 350.7399292, -568.3402710, 385.4368286, -898.2454224, 919.0802002

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8211117, upper bound: 808.8201053
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8211121, upper bound: 808.8365467
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -673.0687256, 546.6938477, -425.1604309, 356.2255859, -1029.2943115, 967.6073608
1: -541.0684204, 531.0290527, -340.4492493, 345.4201355, -886.4885254, 867.5596313
2: -782.8427124, 578.2561035, -494.7025146, 377.2825928, -1160.1252441, 1068.7027588
3: -297.9616089, 757.8487549, -192.4686737, 484.4811096, -778.6732178, 950.3174438
4: -870.5672607, 571.6234131, -551.6278076, 373.9263916, -1244.4936523, 1119.1979980

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0020697, upper bound: 808.8218766
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0020697, upper bound: 809.0027676
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -394.8950500, 334.1777039, -444.8894653, 372.2908936, -767.1859131, 779.0671387
1: -316.1249084, 323.8029480, -356.6080017, 360.9292603, -677.0541992, 680.4108276
2: -459.4748840, 353.5050354, -518.1353760, 393.8309326, -853.3057861, 871.6402588
3: -179.7761841, 450.4103699, -200.9437408, 507.3840332, -687.1600952, 651.3540649
4: -512.8085938, 350.7399292, -577.2812500, 390.2192383, -903.0277100, 928.0211792

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8211572, upper bound: 808.2532531
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8211572, upper bound: 808.2532531
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -673.0687256, 546.6938477, -432.7307129, 361.5239563, -1034.5926514, 975.2204590
1: -541.0684204, 531.0290527, -346.7018127, 350.5121765, -891.5805664, 873.8458862
2: -782.8427124, 578.2561035, -503.5395508, 382.7097778, -1165.5522461, 1077.6135254
3: -297.9616089, 757.8487549, -195.5092926, 492.9059448, -787.1408691, 953.3579712
4: -870.5672607, 571.6234131, -561.1024170, 379.1996460, -1249.7668457, 1128.7567139

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0021153, upper bound: 808.2539096
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0021153, upper bound: 808.5811780
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -400.8190918, 338.1311340, -437.7723999, 367.4412842, -768.2603760, 775.9033813
1: -321.0563354, 327.5602722, -350.6974182, 356.2439270, -677.3002319, 678.2576904
2: -466.4544678, 357.4834900, -509.7792053, 388.8671265, -855.3215942, 867.2626953
3: -182.1852417, 457.0790405, -198.0632477, 499.4380493, -681.6231689, 655.1421509
4: -520.2389526, 354.6555786, -568.3402710, 385.4368286, -905.6757812, 922.9957275

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2532554, upper bound: 808.2810710
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2532554, upper bound: 808.2810710
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -683.3941650, 554.1363525, -425.1604309, 356.2255859, -1039.6197510, 975.2772827
1: -549.4193726, 538.1006470, -340.4492493, 345.4201355, -894.8394775, 874.8026123
2: -794.7759399, 585.9496460, -494.7025146, 377.2825928, -1172.0585938, 1076.5765381
3: -302.0893250, 768.9462280, -192.4686737, 484.4811096, -782.9619141, 961.4148560
4: -883.6091919, 579.2677612, -551.6278076, 373.9263916, -1257.5355225, 1127.0458984

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5804209, upper bound: 808.8220174
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5804209, upper bound: 809.0002849
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -400.8190918, 338.1311340, -444.8894653, 372.2908936, -773.1099854, 783.0205688
1: -321.0563354, 327.5602722, -356.6080017, 360.9292603, -681.9855347, 684.1682129
2: -466.4544678, 357.4834900, -518.1353760, 393.8309326, -860.2854004, 875.6188965
3: -182.1852417, 457.0790405, -200.9437408, 507.3840332, -689.5692139, 658.0227051
4: -520.2389526, 354.6555786, -577.2812500, 390.2192383, -910.4581909, 931.9367676

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2532554, upper bound: 808.2532124
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2532554, upper bound: 808.2641100
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -683.3941650, 554.1363525, -432.7307129, 361.5239563, -1044.9180908, 982.9945068
1: -549.4193726, 538.1006470, -346.7018127, 350.5121765, -899.9315186, 881.1685791
2: -794.7759399, 585.9496460, -503.5395508, 382.7097778, -1177.4855957, 1085.5872803
3: -302.0893250, 768.9462280, -195.5092926, 492.9059448, -791.4971924, 964.4554443
4: -883.6091919, 579.2677612, -561.1024170, 379.1996460, -1262.8085938, 1136.7174072

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5804209, upper bound: 808.2540137
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5804209, upper bound: 808.5812322
time: 0.64 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.14 seconds
NS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.14
Output dim: 4, lower bound: -805.5935406, upper bound: 807.2592837
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -805.5942979, upper bound: 807.8635989
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -805.7842918, upper bound: 807.5131779
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -805.7846018, upper bound: 808.2353429
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -805.8378195, upper bound: 807.9988028
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -805.8378312, upper bound: 807.9959642
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -806.1253426, upper bound: 808.1825999
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -806.2179647, upper bound: 808.1885050
NS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.14
Output dim: 4, lower bound: -807.1904402, upper bound: 804.9128426
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -808.4309983, upper bound: 806.2183171
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -807.9383225, upper bound: 805.8379355
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -808.4182578, upper bound: 806.2183171
NS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.14
Output dim: 4, lower bound: -806.7042999, upper bound: 804.9124045
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -807.8261256, upper bound: 806.2179647
NS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.14
Output dim: 4, lower bound: -807.1529059, upper bound: 805.8371398
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -807.8261256, upper bound: 806.2179647
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -808.8211117, upper bound: 808.8201053
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -808.8211121, upper bound: 808.8365467
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -809.0020697, upper bound: 808.8218766
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -809.0020697, upper bound: 809.0027676
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -808.8211572, upper bound: 808.2532531
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -808.8211572, upper bound: 808.2532531
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -809.0021153, upper bound: 808.2539096
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -809.0021153, upper bound: 808.5811780
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -808.2532554, upper bound: 808.2810710
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -808.2532554, upper bound: 808.2810710
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -808.5804209, upper bound: 808.8220174
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -808.5804209, upper bound: 809.0002849
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -808.2532554, upper bound: 808.2532124
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -808.2532554, upper bound: 808.2641100
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -808.5804209, upper bound: 808.2540137
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.14
Output dim: 4, lower bound: -808.5804209, upper bound: 808.5812322

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -242.1392365, 212.2721100, -490.2859497, 409.0213928, -651.1606445, 702.5580444
1: -192.4254608, 207.5910492, -393.1646118, 396.3913879, -588.8168335, 600.7556763
2: -276.9956970, 227.9431305, -571.1326904, 432.4869690, -709.4825439, 799.0756836
3: -116.1951065, 277.5353394, -221.7794342, 558.2441406, -674.4392700, 499.3147583
4: -310.3807678, 225.1000824, -636.5913086, 429.2048645, -739.5856323, 861.6912842

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5698909, upper bound: 807.8632711
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.3324448, upper bound: 807.8635990
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.3324448, upper bound: 807.8635990
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -332.5247192, 280.1680603, -402.8525085, 341.1595459, -673.6842041, 683.0205688
1: -265.5902405, 273.2568970, -322.5713196, 330.5337524, -596.1240234, 595.8281860
2: -383.7812500, 299.0788269, -469.0149536, 360.7241211, -744.5051270, 768.0937500
3: -152.8173065, 379.1506042, -183.2427368, 460.5935669, -613.4108276, 562.3933105
4: -428.3726501, 295.1049500, -523.3585815, 358.0090332, -786.3815918, 818.4635010

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7565007, upper bound: 807.4679123
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7841265, upper bound: 807.5129517
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -337.4264221, 283.9366150, -505.0977478, 420.0357666, -757.4621582, 789.0342407
1: -269.5048218, 276.9296265, -405.1225281, 407.2164612, -676.7213135, 682.0521240
2: -389.4772339, 303.1282349, -588.5635986, 444.3673096, -833.8444214, 891.6917725
3: -154.7938080, 384.6036682, -227.9976196, 574.9717407, -729.7655640, 612.6013184
4: -434.6884460, 299.0636597, -655.8900146, 440.8576050, -875.5459595, 954.9535522

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7566279, upper bound: 808.2039220
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7842500, upper bound: 808.2353215
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -230.1211853, 202.5869751, -364.0432434, 306.5977478, -536.7189331, 566.6302490
1: -182.7263794, 198.2754822, -291.4953613, 297.3742981, -480.1006775, 489.7707825
2: -262.9144897, 217.8549194, -422.9435120, 324.9241333, -587.8385010, 640.7984619
3: -110.7996674, 263.9035034, -165.2809448, 415.2699280, -526.0695801, 429.1844482
4: -294.6611938, 215.1001282, -470.8506775, 321.7861023, -616.4472656, 685.9508057

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.1406449, upper bound: 806.1426001
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.4057138, upper bound: 806.1426081
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -243.7677917, 213.4292908, -420.5250244, 353.2691040, -597.0368652, 633.9543457
1: -193.7469482, 208.7287445, -336.8798523, 342.4283142, -536.1751099, 545.6085205
2: -278.9052429, 229.1844482, -489.3899231, 373.8004456, -652.7054443, 718.5743408
3: -116.8389282, 279.3439636, -190.6808777, 479.7297363, -596.5686646, 470.0248413
4: -312.4699707, 226.3032074, -545.5039673, 370.5400085, -683.0098877, 771.8070679

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.6876221, upper bound: 807.9704625
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5675451, upper bound: 807.9476717
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -324.0809326, 273.0199890, -380.1496887, 318.2424622, -642.3233643, 653.1696777
1: -258.7407227, 266.4360352, -304.5241394, 308.7597961, -567.5004883, 570.9600830
2: -373.8163147, 291.7880859, -441.9195862, 337.3076477, -711.1239014, 733.7076416
3: -148.6525116, 369.3624878, -171.6442566, 433.2800598, -581.9325562, 541.0067139
4: -417.1934204, 287.7967834, -491.8168945, 333.9521179, -751.1455078, 779.6136475

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.9256579, upper bound: 808.1340797
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8702559, upper bound: 807.9295933
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7842636, upper bound: 808.1659106
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -339.1815186, 285.2177429, -436.6898499, 365.1800842, -704.3615723, 721.9075928
1: -270.9281616, 278.1785583, -349.9454956, 354.0716248, -624.9997559, 628.1240234
2: -391.5458069, 304.4880066, -508.4296265, 386.5229187, -778.0686646, 812.9176025
3: -155.5240631, 386.5900574, -197.2324371, 497.9155884, -653.4395752, 583.8222046
4: -436.9592896, 300.3887939, -566.5614014, 382.8739319, -819.8331909, 866.9501953

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.9498317, upper bound: 808.1539770
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2172850, upper bound: 808.1456405
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -434.0058594, 364.2015076, -332.7971191, 281.7926941, -715.7984619, 696.9985962
1: -347.6188049, 353.1040955, -265.7110901, 274.8519287, -622.4705200, 618.8151855
2: -505.2263184, 385.4602966, -384.1240845, 300.9845886, -806.2109375, 769.5842896
3: -196.3519897, 495.0065613, -153.6954041, 379.7464600, -576.0984497, 648.7019043
4: -563.3115845, 382.0725098, -429.0441895, 297.0279236, -860.3394775, 811.1166382

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.3667908, upper bound: 804.0942061
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.4654827, upper bound: 806.0653839
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4305217, upper bound: 806.2179753
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -418.0475159, 352.3695374, -243.7677917, 213.4292908, -631.4766846, 596.1372681
1: -334.7232056, 341.5417480, -193.7469482, 208.7287445, -543.4518433, 535.2886963
2: -486.4133606, 372.9243774, -278.9052429, 229.1844482, -715.5977783, 651.8295288
3: -189.8340912, 477.0246277, -116.8389282, 279.3439636, -469.1780396, 593.8635254
4: -542.4917603, 369.8348999, -312.4699707, 226.3032074, -768.7949829, 682.3047485

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7555429, upper bound: 805.4328554
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -434.0058594, 364.2015076, -339.1815186, 285.2177429, -719.2236328, 703.3830566
1: -347.6188049, 353.1040955, -270.9281616, 278.1785583, -625.7973633, 624.0322266
2: -505.2263184, 385.4602966, -391.5458069, 304.4880066, -809.7142944, 777.0060425
3: -196.3519897, 495.0065613, -155.5240631, 386.5900574, -582.9419556, 650.5304565
4: -563.3115845, 382.0725098, -436.9592896, 300.3887939, -863.7003784, 819.0317993

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6297972, upper bound: 803.5250241
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -436.6898499, 365.1800842, -332.7971191, 281.7926941, -718.4825439, 697.9771729
1: -349.9454956, 354.0716248, -265.7110901, 274.8519287, -624.7973022, 619.7827148
2: -508.4296265, 386.5229187, -384.1240845, 300.9845886, -809.4141846, 770.6468506
3: -197.2324371, 497.9155884, -153.6954041, 379.7464600, -576.9788208, 651.6109619
4: -566.5614014, 382.8739319, -429.0441895, 297.0279236, -863.5893555, 811.9179688

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5424912, upper bound: 806.2167479
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8249600, upper bound: 806.2172850
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -436.6898499, 365.1800842, -339.1815186, 285.2177429, -721.9075928, 704.3615723
1: -349.9454956, 354.0716248, -270.9281616, 278.1785583, -628.1240234, 624.9997559
2: -508.4296265, 386.5229187, -391.5458069, 304.4880066, -812.9176025, 778.0686646
3: -197.2324371, 497.9155884, -155.5240631, 386.5900574, -583.8222046, 653.4395752
4: -566.5614014, 382.8739319, -436.9592896, 300.3887939, -866.9501953, 819.8331909

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.7838145, upper bound: 805.7832223
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.3305539, upper bound: 805.7840165
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -394.8950500, 334.1777039, -394.8950500, 334.1777039, -729.0727539, 729.0727539
1: -316.1249084, 323.8029480, -316.1249084, 323.8029480, -639.9278564, 639.9278564
2: -459.4748840, 353.5050354, -459.4748840, 353.5050354, -812.9797974, 812.9797363
3: -179.7761841, 450.4103699, -179.7761841, 450.4103699, -630.1864624, 630.1864624
4: -512.8085938, 350.7399292, -512.8085938, 350.7399292, -863.5485229, 863.5485229

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1690196, upper bound: 808.4939443
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6970602, upper bound: 808.6956470
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -394.8950500, 334.1777039, -672.6865234, 546.3706665, -936.9903564, 1006.8642578
1: -316.1249084, 323.8029480, -540.7593994, 530.7145386, -842.8943481, 864.5621338
2: -459.4748840, 353.5050354, -782.3930664, 577.9171143, -1033.0053711, 1135.8980713
3: -179.7761841, 450.4103699, -297.7907410, 757.4197388, -937.1958618, 744.2760620
4: -512.8085938, 350.7399292, -870.0671997, 571.2935181, -1079.9075928, 1220.8071289

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1690201, upper bound: 808.4939443
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6970602, upper bound: 808.7492196
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -672.6865234, 546.3706665, -394.8949280, 334.1775818, -1006.8640747, 936.9901733
1: -540.7593994, 530.7145386, -316.1248169, 323.8027649, -864.5620728, 842.8942261
2: -782.3930664, 577.9171143, -459.4747009, 353.5049133, -1135.8979492, 1033.0051270
3: -297.7907410, 757.4197388, -179.7761078, 450.4101868, -744.2758179, 937.1958008
4: -870.0671997, 571.2935181, -512.8084106, 350.7398071, -1220.8070068, 1079.9073486

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9345310, upper bound: 808.8214705
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9884991, upper bound: 808.8212992
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -673.0687256, 546.6938477, -673.0441895, 546.6687622, -1210.4461670, 1210.4445801
1: -541.0684204, 531.0290527, -541.0486450, 531.0049438, -1064.6021729, 1064.6053467
2: -782.8427124, 578.2561035, -782.8121948, 578.2310181, -1352.5545654, 1352.5474854
3: -297.9616089, 757.8487549, -297.9514771, 757.8125610, -1050.3708496, 1050.3964844
4: -870.5672607, 571.6234131, -870.5338745, 571.5978394, -1432.8653564, 1432.8558350

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9345310, upper bound: 808.9889000
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9884991, upper bound: 808.9887533
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -394.8950500, 334.1777039, -400.8190918, 338.1311340, -733.0261841, 734.9968262
1: -316.1249084, 323.8029480, -321.0563354, 327.5602722, -643.6851807, 644.8591309
2: -459.4748840, 353.5050354, -466.4544678, 357.4834900, -816.9583740, 819.9594116
3: -179.7761841, 450.4103699, -182.1852417, 457.0790405, -636.8549805, 632.5954590
4: -512.8085938, 350.7399292, -520.2389526, 354.6555786, -867.4639893, 870.9788818

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1690848, upper bound: 808.2320062
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6971254, upper bound: 808.2325049
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -394.8950500, 334.1777039, -683.0560913, 553.8505859, -944.6949463, 1017.2337646
1: -316.1249084, 323.8029480, -549.1461182, 537.8226318, -850.1716919, 872.9489746
2: -459.4748840, 353.5050354, -794.3785400, 585.6498413, -1040.9151611, 1147.8835449
3: -179.7761841, 450.4103699, -301.9382324, 768.5673828, -948.3434448, 748.5824585
4: -512.8085938, 350.7399292, -883.1671143, 578.9761963, -1087.7910156, 1233.9068604

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1690848, upper bound: 808.3507056
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6971254, upper bound: 808.4941811
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -672.6678467, 546.3547974, -400.8190918, 338.1311340, -1010.7989502, 942.9947510
1: -540.7442627, 530.6991577, -321.0563354, 327.5602722, -868.3044434, 847.8828125
2: -782.3709717, 577.9004517, -466.4544678, 357.4834900, -1139.8544922, 1040.1174316
3: -297.7824097, 757.3986816, -182.1852417, 457.0790405, -751.0068359, 939.5838623
4: -870.0428467, 571.2774048, -520.2389526, 354.6555786, -1224.6983643, 1087.4892578

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9345601, upper bound: 808.2535603
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9885326, upper bound: 808.2533992
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -673.0687256, 546.6938477, -683.3704834, 554.1118774, -1218.1164551, 1220.2379150
1: -541.0684204, 531.0290527, -549.4003906, 538.0770874, -1071.8458252, 1072.5794678
2: -782.8427124, 578.2561035, -794.7464600, 585.9254150, -1360.4285889, 1363.9803467
3: -297.9616089, 757.8487549, -302.0794983, 768.9109497, -1061.3103027, 1054.6855469
4: -870.5672607, 571.6234131, -883.5769653, 579.2427979, -1440.7141113, 1445.3806152

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9345601, upper bound: 808.3419601
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9885326, upper bound: 808.3417853
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -400.8190918, 338.1311340, -394.8950500, 334.1777039, -734.9968262, 733.0261841
1: -321.0563354, 327.5602722, -316.1249084, 323.8029480, -644.8590698, 643.6851807
2: -466.4544678, 357.4834900, -459.4748840, 353.5050354, -819.9594116, 816.9583740
3: -182.1852417, 457.0790405, -179.7761841, 450.4103699, -632.5955200, 636.8549805
4: -520.2389526, 354.6555786, -512.8085938, 350.7399292, -870.9788818, 867.4640503

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -400.8190918, 338.1311340, -672.6678467, 546.3547974, -942.9947510, 1010.7988892
1: -321.0563354, 327.5602722, -540.7442627, 530.6991577, -847.8827515, 868.3044434
2: -466.4544678, 357.4834900, -782.3709717, 577.9004517, -1040.1175537, 1139.8544922
3: -182.1852417, 457.0790405, -297.7824097, 757.3986816, -939.5839233, 751.0068359
4: -520.2389526, 354.6555786, -870.0428467, 571.2774048, -1087.4891357, 1224.6982422

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -683.0560913, 553.8505859, -394.8949280, 334.1775818, -1017.2336426, 944.6948242
1: -549.1461182, 537.8226318, -316.1248169, 323.8027649, -872.9488525, 850.1715698
2: -794.3785400, 585.6498413, -459.4747009, 353.5049133, -1147.8834229, 1040.9150391
3: -301.9382324, 768.5673828, -179.7761078, 450.4101868, -748.5822144, 948.3433838
4: -883.1671143, 578.9761963, -512.8084106, 350.7398071, -1233.9068604, 1087.7908936

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5461097, upper bound: 808.8218318
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3415222, upper bound: 808.5260064
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -683.3941650, 554.1363525, -673.0441895, 546.6687622, -1220.2384033, 1218.1145020
1: -549.4193726, 538.1006470, -541.0486450, 531.0049438, -1072.5754395, 1071.8481445
2: -794.7759399, 585.9496460, -782.8121948, 578.2310181, -1363.9862061, 1360.4211426
3: -302.0893250, 768.9462280, -297.9514771, 757.8125610, -1054.6594238, 1061.3347168
4: -883.6091919, 579.2677612, -870.5338745, 571.5978394, -1445.3890381, 1440.7039795

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5461097, upper bound: 808.9891299
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3415222, upper bound: 808.5259964
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -400.8190918, 338.1311340, -400.8190918, 338.1311340, -738.9501953, 738.9501953
1: -321.0563354, 327.5602722, -321.0563354, 327.5602722, -648.6164551, 648.6164551
2: -466.4544678, 357.4834900, -466.4544678, 357.4834900, -823.9379883, 823.9379883
3: -182.1852417, 457.0790405, -182.1852417, 457.0790405, -639.2640381, 639.2640381
4: -520.2389526, 354.6555786, -520.2389526, 354.6555786, -874.8943481, 874.8944092

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0667095, upper bound: 808.0956122
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2531365, upper bound: 808.2530871
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -400.8190918, 338.1311340, -683.0380859, 553.8354492, -950.8041992, 1021.1691284
1: -321.0563354, 327.5602722, -549.1316528, 537.8078003, -855.2402344, 876.6918335
2: -466.4544678, 357.4834900, -794.3571777, 585.6339111, -1048.1284180, 1151.8406982
3: -182.1852417, 457.0790405, -301.9301453, 768.5472412, -950.7323608, 755.3813477
4: -520.2389526, 354.6555786, -883.1436157, 578.9607544, -1095.4862061, 1237.7990723

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0667095, upper bound: 808.0956230
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2531365, upper bound: 808.2639853
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -683.0380859, 553.8354492, -400.8190918, 338.1311340, -1021.1691895, 950.8041382
1: -549.1316528, 537.8078003, -321.0563354, 327.5602722, -876.6918335, 855.2402344
2: -794.3571777, 585.6339111, -466.4544678, 357.4834900, -1151.8406982, 1048.1284180
3: -301.9301453, 768.5472412, -182.1852417, 457.0790405, -755.3813477, 950.7323608
4: -883.1436157, 578.9607544, -520.2389526, 354.6555786, -1237.7990723, 1095.4860840

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5461027, upper bound: 808.2538210
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3415157, upper bound: 808.2532812
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -683.3941650, 554.1363525, -683.3704834, 554.1118774, -1228.0128174, 1228.0118408
1: -549.4193726, 538.1006470, -549.4003906, 538.0770874, -1079.8988037, 1079.9017334
2: -794.7759399, 585.9496460, -794.7464600, 585.9254150, -1371.9604492, 1371.9539795
3: -302.0893250, 768.9462280, -302.0794983, 768.9109497, -1065.6663818, 1065.6910400
4: -883.6091919, 579.2677612, -883.5769653, 579.2427979, -1453.3504639, 1453.3414307

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5461027, upper bound: 808.3423157
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3415157, upper bound: 808.3416668
time: 0.59 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.67 seconds
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -805.3324448, upper bound: 807.8635990
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -805.3324448, upper bound: 807.8635990
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -805.7565007, upper bound: 807.4679123
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -805.7841265, upper bound: 807.5129517
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -805.7566279, upper bound: 808.2039220
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -805.7842500, upper bound: 808.2353215
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.67
Output dim: 4, lower bound: -805.1406449, upper bound: 806.1426001
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.67
Output dim: 4, lower bound: -805.4057138, upper bound: 806.1426081
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -805.6876221, upper bound: 807.9704625
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -805.5675451, upper bound: 807.9476717
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -805.8702559, upper bound: 807.9295933
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -805.7842636, upper bound: 808.1659106
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -805.9498317, upper bound: 808.1539770
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -806.2172850, upper bound: 808.1456405
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.67
Output dim: 4, lower bound: -805.4654827, upper bound: 806.0653839
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.4305217, upper bound: 806.2179753
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -807.5424912, upper bound: 806.2167479
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -807.8249600, upper bound: 806.2172850
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.67
Output dim: 4, lower bound: -805.7838145, upper bound: 805.7832223
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.67
Output dim: 4, lower bound: -807.3305539, upper bound: 805.7840165
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.1690196, upper bound: 808.4939443
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.6970602, upper bound: 808.6956470
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.1690201, upper bound: 808.4939443
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.6970602, upper bound: 808.7492196
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.9345310, upper bound: 808.8214705
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.9884991, upper bound: 808.8212992
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.9345310, upper bound: 808.9889000
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.9884991, upper bound: 808.9887533
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.1690848, upper bound: 808.2320062
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.6971254, upper bound: 808.2325049
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.1690848, upper bound: 808.3507056
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.6971254, upper bound: 808.4941811
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.9345601, upper bound: 808.2535603
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.9885326, upper bound: 808.2533992
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.9345601, upper bound: 808.3419601
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.9885326, upper bound: 808.3417853
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.5461097, upper bound: 808.8218318
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.3415222, upper bound: 808.5260064
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.5461097, upper bound: 808.9891299
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.3415222, upper bound: 808.5259964
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.0667095, upper bound: 808.0956122
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.2531365, upper bound: 808.2530871
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.0667095, upper bound: 808.0956230
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.2531365, upper bound: 808.2639853
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.5461027, upper bound: 808.2538210
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.3415157, upper bound: 808.2532812
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.5461027, upper bound: 808.3423157
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 4, lower bound: -808.3415157, upper bound: 808.3416668

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -216.2514648, 192.2908783, -490.2859497, 409.0213928, -625.2728271, 682.5768433
1: -171.6279144, 188.0630341, -393.1646118, 396.3913879, -568.0192261, 581.2276611
2: -246.7999725, 206.4940338, -571.1326904, 432.4869690, -679.2869263, 777.6267090
3: -105.3072586, 248.2400818, -221.7794342, 558.2441406, -663.5513306, 470.0195312
4: -276.9530334, 204.1986389, -636.5913086, 429.2048645, -706.1578979, 840.7899170

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -804.5065838, upper bound: 807.4577569
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -325.3053284, 275.1112061, -490.2859497, 409.0213928, -734.3267212, 765.3971558
1: -259.4033203, 267.5791931, -393.1646118, 396.3913879, -655.7946777, 660.7437744
2: -373.8045959, 293.0217590, -571.1326904, 432.4869690, -806.2915649, 864.1544189
3: -151.9472504, 369.8461304, -221.7794342, 558.2441406, -710.1914062, 591.6255493
4: -417.9171143, 290.3411560, -636.5913086, 429.2048645, -847.1219482, 926.9324951

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.5065838, upper bound: 807.2239462
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -202.2097473, 182.9089813, -380.5158691, 325.1861267, -527.3958130, 563.4248657
1: -160.8273773, 178.8341522, -304.5064697, 315.0050659, -475.8324585, 483.3406372
2: -232.4548950, 196.8733826, -442.7926025, 343.9697266, -576.4244385, 639.6660156
3: -99.2597504, 235.3741150, -174.4057312, 435.9344482, -535.1941528, 409.7798462
4: -260.1893921, 194.0823059, -494.2933655, 341.4176331, -601.6070557, 688.3756714

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -319.4223328, 270.3036499, -401.1340637, 339.8118591, -659.2341919, 671.4377441
1: -255.1098785, 263.9116211, -321.1917114, 329.2502441, -584.3601074, 585.1033325
2: -368.3774719, 288.9244080, -466.9825439, 359.3422546, -727.7196045, 755.9069824
3: -147.8404541, 365.1464233, -182.5261993, 458.6691895, -606.5096436, 547.6725464
4: -411.3557434, 284.9787292, -521.1022339, 356.6153870, -767.9711304, 806.0809326

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -207.0722809, 186.5493774, -481.8200378, 403.0580750, -610.1303711, 668.3692627
1: -164.7147064, 182.3938904, -386.2932739, 390.6610718, -555.3757324, 568.6871338
2: -238.0717621, 200.7930145, -561.2456665, 426.3619690, -664.4337158, 762.0386963
3: -101.2839813, 240.7201385, -218.2745972, 549.0928345, -650.3767090, 458.9947510
4: -266.4134827, 197.9013824, -625.6207275, 422.9972534, -689.4107056, 823.5219727

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -323.6615295, 273.6445923, -503.6019897, 418.7790833, -742.4406128, 777.2465820
1: -258.4906616, 267.1634827, -403.9259338, 406.0364075, -664.5270996, 671.0892944
2: -373.2900391, 292.5292358, -586.7998657, 443.0890198, -816.3790283, 879.3290405
3: -149.5711212, 369.8964233, -227.3803711, 573.3015747, -722.8726807, 597.2766113
4: -416.8058472, 288.5038452, -653.9299927, 439.5772095, -856.3830566, 942.4338379

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -128.8563995, 131.8969574, -391.8698425, 331.3397522, -460.1961670, 523.7667236
1: -101.6041031, 129.2213898, -313.6405334, 321.2483826, -422.8524780, 442.8619385
2: -146.0759277, 143.4796143, -455.5387878, 351.0606995, -497.1366272, 599.0183105
3: -72.2526550, 154.9948578, -179.0797729, 447.5369873, -519.7896118, 334.0745850
4: -164.7269440, 141.6687927, -507.9393005, 347.7798157, -512.5067139, 649.6080322

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.5695484, upper bound: 806.8389001
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5696926, upper bound: 807.9698564
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -234.4331360, 207.0217896, -415.8563538, 349.1046753, -583.5377808, 622.8781738
1: -186.2812653, 202.6648712, -333.0870361, 338.4895325, -524.7708130, 535.7518921
2: -267.8715210, 222.6802063, -483.7678223, 369.7138367, -637.5853271, 706.4479980
3: -113.3358078, 269.6219482, -188.7321472, 474.2912903, -587.6270752, 458.3540955
4: -300.2645264, 219.8254852, -539.2694092, 366.2949829, -666.5595093, 759.0948486

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5670282, upper bound: 807.7843779
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5670282, upper bound: 807.9476717
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -292.2451172, 248.5405273, -373.9132080, 313.6596069, -605.9046021, 622.4536743
1: -233.1657715, 242.5902557, -299.5379944, 304.2880554, -537.4538574, 542.1281128
2: -336.5928955, 265.5570679, -434.6731262, 332.3753357, -668.9681396, 700.2301636
3: -135.4245605, 333.5204773, -169.1046906, 426.5419312, -561.9664917, 502.6251526
4: -376.0489502, 262.1647644, -483.7954712, 329.1644287, -705.2133789, 745.9602051

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -390.3752441, 325.4432068, -376.2824707, 315.3312683, -705.7064209, 701.7256470
1: -312.2857666, 316.3437500, -301.3814697, 305.9187317, -618.2044067, 617.7252197
2: -451.3416443, 345.8220825, -437.3587952, 334.2061462, -785.5477295, 783.1808472
3: -178.4863434, 444.1207275, -170.0602264, 428.9143066, -607.4004517, 614.1809692
4: -503.4942627, 342.2582703, -486.8175354, 330.9089661, -834.4031982, 829.0758057

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -208.9084625, 187.8605499, -413.2079468, 348.0126038, -556.9210815, 601.0683594
1: -166.1991119, 183.6752319, -330.9530334, 337.3805237, -503.5795898, 514.6282349
2: -240.2212067, 202.1890564, -480.9251404, 368.4283752, -608.6494141, 683.1141968
3: -102.0045395, 242.7932129, -187.7321167, 471.8979187, -573.9023438, 430.5253296
4: -268.7749023, 199.2576904, -536.0037842, 365.0384521, -633.8132935, 735.2613525

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.7559971, upper bound: 806.8390960
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7561413, upper bound: 808.1535664
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -325.5486145, 275.0176086, -435.2336731, 363.9646606, -689.5133057, 710.2512817
1: -260.0195007, 268.5013733, -348.7747192, 352.9122620, -612.9317627, 617.2761230
2: -375.5151062, 293.9867554, -506.7004395, 385.2971191, -760.8121338, 800.6871338
3: -150.3585815, 372.0249329, -196.5973663, 496.2656860, -646.6242676, 568.6222534
4: -419.2492676, 289.9257812, -564.6414185, 381.6427917, -800.8919678, 854.5671997

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.7835746, upper bound: 806.8391885
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7837188, upper bound: 808.1451800
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -433.9595032, 364.1676025, -332.7971191, 281.7926941, -715.7520752, 696.9646606
1: -347.5808716, 353.0709534, -265.7110901, 274.8519287, -622.4327393, 618.7820435
2: -505.1716614, 385.4242249, -384.1240845, 300.9845886, -806.1561890, 769.5482178
3: -196.3329315, 494.9554138, -153.6954041, 379.7464600, -576.0794067, 648.6508179
4: -563.2519531, 382.0372009, -429.0441895, 297.0279236, -860.2799072, 811.0813599

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.4901762, upper bound: 805.0730347
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.4901763, upper bound: 806.2179753
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -301.9802551, 266.9254761, -310.4858704, 265.7406006, -567.7208252, 577.4112549
1: -241.3724365, 258.4585876, -247.7197571, 259.3047180, -500.6771240, 506.1782227
2: -351.4915771, 282.9038696, -358.0902100, 284.2802124, -635.7716064, 640.9939575
3: -142.9042206, 349.7152405, -144.4660034, 355.2352295, -498.1394348, 494.1812439
4: -392.2167969, 280.8281860, -400.1031189, 280.4832458, -672.6998901, 680.9312744

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -425.2669678, 355.6134644, -331.2540894, 280.6394653, -705.9064331, 686.8675537
1: -340.7726135, 345.0171204, -264.4758911, 273.7566223, -614.5291138, 609.4929199
2: -494.8684692, 376.8844604, -382.3187866, 299.7967224, -794.6651611, 759.2032471
3: -192.2481842, 485.1243896, -153.1095428, 378.0964966, -570.3446655, 638.2338257
4: -551.5020752, 373.2102356, -427.0529175, 295.8470154, -847.3491211, 800.2631836

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.5346103, upper bound: 805.0730165
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8189308, upper bound: 806.2172850
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -366.3092041, 313.3845520, -388.7923889, 329.8008118, -696.1099854, 702.1769409
1: -293.1076965, 303.4456177, -311.2268982, 319.5110168, -612.6187134, 614.6724854
2: -426.1185608, 331.2268677, -452.3634644, 348.8051758, -774.9237061, 783.5903320
3: -168.1203613, 419.2138367, -177.2756500, 443.7620544, -611.8823242, 596.4892578
4: -475.9714966, 329.0264893, -504.9484558, 346.1679993, -822.1394653, 833.9749756

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1686534, upper bound: 808.1682240
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1686534, upper bound: 808.4946725
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -464.3193970, 387.3415527, -392.5695801, 332.3095398, -796.6289062, 779.9110718
1: -372.3395691, 375.2162781, -314.2249451, 321.9737549, -694.3133545, 689.4412231
2: -540.9116211, 409.2961121, -456.7168274, 351.5197449, -892.4312744, 866.0129395
3: -210.2126617, 528.4973145, -178.7674408, 447.7006531, -657.9133301, 707.2647705
4: -603.1533203, 406.5039062, -509.7803345, 348.7909851, -951.9443359, 916.2842407

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6966942, upper bound: 808.1689398
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6966942, upper bound: 808.6956490
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -366.3092041, 313.3845520, -665.8491211, 540.8678589, -902.4321899, 979.2336426
1: -293.1076965, 303.4456177, -535.2368164, 525.3574829, -814.1109009, 838.6824341
2: -426.1185608, 331.2268677, -774.3309326, 572.0941162, -993.1840210, 1105.5578613
3: -168.1203613, 419.2138367, -294.8558350, 749.6275024, -917.7478027, 709.6298828
4: -475.9714966, 329.0264893, -861.1741943, 565.5187378, -1036.6035156, 1190.2005615

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -464.3193970, 387.3415527, -670.6345215, 544.7572021, -1005.1935425, 1057.9760742
1: -372.3395691, 375.2162781, -539.0972900, 529.1295776, -897.6784668, 914.3135986
2: -540.9116211, 409.2961121, -779.9912109, 576.1989746, -1112.8853760, 1189.2872314
3: -210.2126617, 528.4973145, -296.9393005, 755.0941772, -965.3068237, 821.5428467
4: -603.1533203, 406.5039062, -867.4188232, 569.6449585, -1168.8857422, 1273.9227295

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6971277, upper bound: 808.4305106
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6972924, upper bound: 808.7492118
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -585.6378784, 475.3216553, -375.8108521, 318.6907349, -904.3284912, 845.9268188
1: -470.4942627, 461.9383240, -300.6746826, 308.8512573, -779.3455200, 757.9031982
2: -680.1019287, 503.5285950, -436.9044495, 337.2741699, -1017.3760986, 935.1022949
3: -259.8985291, 657.0894775, -171.3284760, 428.5075378, -683.7975464, 828.4179688
4: -756.0987549, 497.5861206, -487.7096863, 334.5729675, -1090.6716309, 980.1320801

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.7122771, upper bound: 808.7956048
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -666.3740845, 540.6988525, -394.8949280, 334.1775818, -1000.5515747, 931.7448120
1: -535.6275024, 525.2710571, -316.1248169, 323.8027649, -859.4302368, 837.7927246
2: -774.9025879, 572.0125122, -459.4747009, 353.5049133, -1128.4074707, 1027.5335693
3: -294.7211914, 750.0682373, -179.7761078, 450.4101868, -741.5384521, 929.8442993
4: -861.8061523, 565.3389893, -512.8084106, 350.7398071, -1212.5458984, 1074.4224854

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.6796580, upper bound: 808.7954033
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9626814, upper bound: 805.5032704
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9881497, upper bound: 808.8207930
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -586.0347900, 475.6571045, -652.2883301, 529.8769531, -1105.6870117, 1117.9068604
1: -470.8153687, 462.2648315, -524.2689209, 514.7816162, -977.2982788, 978.4293213
2: -680.5691528, 503.8807678, -758.3925171, 560.6702881, -1231.8416748, 1253.0167236
3: -260.0764771, 657.5344238, -288.9270325, 733.9650269, -987.8848877, 940.5769653
4: -756.6181641, 497.9292908, -843.3605957, 554.2161865, -1300.7993164, 1331.2216797

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9348007, upper bound: 808.9345821
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9348007, upper bound: 808.9887533
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -666.8051758, 541.0629883, -673.0441895, 546.6687622, -1204.2506104, 1205.2363281
1: -535.9760132, 525.6253662, -541.0486450, 531.0049438, -1059.5368652, 1059.5400391
2: -775.4098511, 572.3945923, -782.8121948, 578.2310181, -1345.1911621, 1347.1151123
3: -294.9136658, 750.5516357, -297.9514771, 757.8125610, -1047.6527100, 1043.1629639
4: -862.3701172, 565.7105103, -870.5338745, 571.5978394, -1424.7893066, 1427.4082031

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9887463, upper bound: 808.9345820
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9887463, upper bound: 808.9887533
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -366.3092041, 313.3845520, -394.9259644, 333.8977356, -700.2069092, 708.3105469
1: -293.1076965, 303.4456177, -316.3163757, 323.4127502, -616.5203247, 619.7619629
2: -426.1185608, 331.2268677, -459.5728149, 352.9360962, -779.0545044, 790.7996826
3: -168.1203613, 419.2138367, -179.7510376, 450.6892090, -618.8093262, 598.9647217
4: -475.9714966, 329.0264893, -512.6291504, 350.2423706, -826.2138672, 841.6556396

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1687985, upper bound: 808.1640583
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1687946, upper bound: 808.2320062
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -464.3193970, 387.3415527, -398.3358154, 336.1829529, -800.5023193, 785.6773682
1: -372.3395691, 375.2162781, -319.0241699, 325.6474609, -697.9870605, 694.2404785
2: -540.9116211, 409.2961121, -463.5128784, 355.4131165, -896.3247070, 872.8089600
3: -210.2126617, 528.4973145, -181.1436310, 454.2023926, -664.4150391, 709.6408691
4: -603.1533203, 406.5039062, -517.0211792, 352.6264343, -955.7797852, 923.5250244

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6968391, upper bound: 808.1643714
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6968351, upper bound: 808.2323405
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -366.3092041, 313.3845520, -676.0756836, 548.2385864, -910.0352173, 989.4602051
1: -293.1076965, 303.4456177, -543.5075073, 532.3558350, -821.2846680, 846.9531250
2: -426.1185608, 331.2268677, -786.1398315, 579.7113647, -1000.9861450, 1117.3665771
3: -168.1203613, 419.2138367, -298.9448853, 760.6139526, -928.7341919, 713.8826904
4: -475.9714966, 329.0264893, -874.0837402, 573.0966797, -1044.3908691, 1203.1102295

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1691000, upper bound: 808.3131219
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1690153, upper bound: 808.3268553
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -464.3193970, 387.3415527, -680.8857422, 552.1427612, -1012.7982788, 1068.2271729
1: -372.3395691, 375.2162781, -547.3849487, 536.1475830, -904.8615112, 922.6011963
2: -540.9116211, 409.2961121, -791.8348389, 583.8340454, -1120.6925049, 1201.1308594
3: -210.2126617, 528.4973145, -301.0364685, 766.1036987, -976.3163452, 825.7952881
4: -603.1533203, 406.5039062, -880.3670044, 577.2287598, -1176.6643066, 1286.8708496

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.1549632, upper bound: 805.8016789
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6788291, upper bound: 808.3842920
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -585.6190186, 475.3056946, -382.3218689, 323.0285645, -908.6475830, 852.5286255
1: -470.4789429, 461.9228516, -306.0694885, 312.9815369, -783.4603882, 763.3614502
2: -680.0797119, 503.5118103, -444.5477905, 341.6568298, -1021.7365723, 942.8826294
3: -259.8901062, 657.0682373, -173.9769287, 435.7745361, -691.1300049, 831.0451050
4: -756.0739746, 497.5699463, -495.8852539, 338.8803406, -1094.9543457, 988.4649658

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9345625, upper bound: 808.0669425
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9345625, upper bound: 808.2533992
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -666.3551025, 540.6828003, -400.8190918, 338.1311340, -1004.4862061, 937.7491455
1: -535.6122437, 525.2554321, -321.0563354, 327.5602722, -863.1724243, 842.7808838
2: -774.8803711, 571.9957275, -466.4544678, 357.4834900, -1132.3636475, 1034.6457520
3: -294.7127075, 750.0469360, -182.1852417, 457.0790405, -748.2695923, 932.2320557
4: -861.7812500, 565.3226929, -520.2389526, 354.6555786, -1216.4366455, 1082.0039062

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9885350, upper bound: 808.0669425
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9885350, upper bound: 808.2533992
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -586.0347900, 475.6571045, -662.8634033, 537.5527344, -1113.5725098, 1127.9445801
1: -470.8153687, 462.2648315, -532.8203125, 522.0766602, -984.7503052, 986.6002808
2: -680.5691528, 503.8807678, -770.6417847, 568.6123047, -1239.9417725, 1264.7395020
3: -260.0764771, 657.5344238, -293.1807556, 745.3524780, -999.1049805, 944.9809570
4: -756.6181641, 497.9292908, -856.7531128, 562.0997314, -1308.8608398, 1344.0241699

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9346769, upper bound: 808.3417853
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9346769, upper bound: 808.3417853
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -666.8051758, 541.0629883, -683.3704834, 554.1118774, -1211.9210205, 1215.0294189
1: -535.9760132, 525.6253662, -549.4003906, 538.0770874, -1066.7805176, 1067.5141602
2: -775.4098511, 572.3945923, -794.7464600, 585.9254150, -1353.0653076, 1358.5478516
3: -294.9136658, 750.5516357, -302.0794983, 768.9109497, -1058.5920410, 1047.4520264
4: -862.3701172, 565.7105103, -883.5769653, 579.2427979, -1432.6380615, 1439.9329834

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9886452, upper bound: 808.3417853
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9886452, upper bound: 808.3417853
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -598.8077393, 485.0388184, -375.8108521, 318.6907349, -917.4984741, 855.8459473
1: -481.1808472, 471.1946411, -300.6746826, 308.8512573, -790.0321045, 767.3159180
2: -695.5374146, 513.5653076, -436.9044495, 337.2741699, -1032.8115234, 945.3032837
3: -265.2058716, 671.5277100, -171.3284760, 428.5075378, -689.2772217, 842.8562012
4: -772.9429932, 507.5979309, -487.7096863, 334.5729675, -1107.5159912, 990.3001099

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.6783520, upper bound: 807.8990007
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.7291765, upper bound: 808.7959701
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2364300, upper bound: 805.5032812
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5458523, upper bound: 808.8213256
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -676.5706177, 548.0371094, -394.8949280, 334.1775818, -1010.7481689, 939.3085327
1: -543.8797607, 532.2193604, -316.1248169, 323.8027649, -867.6824951, 844.9118042
2: -786.6802979, 579.5935059, -459.4747009, 353.5049133, -1140.1850586, 1035.2963867
3: -298.7709961, 761.0651855, -179.7761078, 450.4101868, -745.7555542, 940.8411865
4: -874.6568604, 572.8759766, -512.8084106, 350.7398071, -1225.3967285, 1082.1649170

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3146749, upper bound: 805.5032693
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3413173, upper bound: 808.5533870
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -599.1633301, 485.3391724, -652.2883301, 529.8769531, -1118.3315430, 1127.7941895
1: -481.4682617, 471.4866333, -524.2689209, 514.7816162, -987.6015625, 987.8101807
2: -695.9554443, 513.8804932, -758.3925171, 560.6702881, -1246.6738281, 1263.1837158
3: -265.3652039, 671.9259033, -288.9270325, 733.9650269, -993.3485718, 954.7996216
4: -773.4079590, 507.9051819, -843.3605957, 554.2161865, -1316.9825439, 1341.3596191

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3407614, upper bound: 808.3695675
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3407614, upper bound: 808.5260064
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -676.9556274, 548.3621216, -673.0441895, 546.6687622, -1213.9038086, 1212.7637939
1: -544.1911011, 532.5355835, -541.0486450, 531.0049438, -1067.3950195, 1066.6236572
2: -787.1326904, 579.9343872, -782.8121948, 578.2310181, -1356.4234619, 1354.8395996
3: -298.9427490, 761.4962158, -297.9514771, 757.8125610, -1051.8509521, 1053.9503174
4: -875.1602783, 573.2073975, -870.5338745, 571.5978394, -1437.0405273, 1435.1137695

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3407611, upper bound: 808.3695643
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3407611, upper bound: 808.5259964
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -348.8487854, 294.2696228, -382.3218689, 323.0285645, -671.8773193, 676.5913086
1: -279.3065796, 285.4207153, -306.0694885, 312.9815369, -592.2880859, 591.4901123
2: -405.0474854, 311.6168823, -444.5477905, 341.6568298, -746.7042236, 756.1646118
3: -158.3417816, 397.9795532, -173.9769287, 435.7745361, -594.1160889, 571.9564209
4: -451.2043457, 308.8600769, -495.8852539, 338.8803406, -790.0845947, 804.7451782

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0657093, upper bound: 808.0657259
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0657092, upper bound: 808.1704936
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -392.7154541, 331.1437073, -400.8190918, 338.1311340, -730.8465576, 731.9627686
1: -314.4709778, 320.8226929, -321.0563354, 327.5602722, -642.0311890, 641.8788452
2: -456.8787231, 350.2012329, -466.4544678, 357.4834900, -814.3621826, 816.6557007
3: -178.5253296, 447.7351379, -182.1852417, 457.0790405, -635.6042480, 629.9202881
4: -509.6759033, 347.4036255, -520.2389526, 354.6555786, -864.3314209, 867.6425781

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0660128, upper bound: 808.0666322
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0660128, upper bound: 808.2530871
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -348.8487854, 294.2696228, -662.4821167, 537.2338867, -880.6912231, 956.7515869
1: -279.3065796, 285.4207153, -532.5122681, 521.7661743, -796.2312622, 817.9329834
2: -405.0474854, 311.6168823, -770.1951294, 568.2766113, -967.8133545, 1081.8116455
3: -158.3417816, 397.9795532, -293.0096130, 744.9324341, -903.2740479, 686.3536377
4: -451.2043457, 308.8600769, -856.2557983, 561.7742310, -1007.5471802, 1165.1156006

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0661778, upper bound: 808.0802980
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0661778, upper bound: 808.0956122
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -392.7154541, 331.1437073, -682.9857178, 553.7911987, -942.1362915, 1014.1293945
1: -314.4709778, 320.8226929, -549.0891113, 537.7646484, -848.1856079, 869.9117432
2: -456.8787231, 350.2012329, -794.2956543, 585.5875244, -1037.8353271, 1144.4965820
3: -178.5253296, 447.7351379, -301.9067383, 768.4886475, -947.0139160, 745.7413330
4: -509.6759033, 347.4036255, -883.0750732, 578.9155884, -1084.1906738, 1230.4787598

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0762676, upper bound: 808.0889792
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0762758, upper bound: 808.2639853
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -598.7894287, 485.0233765, -382.3218689, 323.0285645, -921.8179932, 862.5463867
1: -481.1660156, 471.1796875, -306.0694885, 312.9815369, -794.1475220, 772.8502197
2: -695.5159302, 513.5491333, -444.5477905, 341.6568298, -1037.1727295, 953.1797485
3: -265.1976624, 671.5072021, -173.9769287, 435.7745361, -696.6752319, 845.4839478
4: -772.9189453, 507.5821838, -495.8852539, 338.8803406, -1111.7993164, 998.7404175

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0664548, upper bound: 808.0667456
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0664548, upper bound: 808.2532812
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -676.5523682, 548.0217896, -400.8190918, 338.1311340, -1014.6834717, 945.4172363
1: -543.8651733, 532.2043457, -321.0563354, 327.5602722, -871.4254150, 849.9802246
2: -786.6586304, 579.5773315, -466.4544678, 357.4834900, -1144.1420898, 1042.5090332
3: -298.7628784, 761.0447998, -182.1852417, 457.0790405, -752.5544434, 943.2299194
4: -874.6330566, 572.8602295, -520.2389526, 354.6555786, -1229.2885742, 1089.8593750

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0664588, upper bound: 808.0667456
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0664588, upper bound: 808.2532812
time: 0.68 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.54 + 416.54 = 420.08 seconds
