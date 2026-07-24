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
execution time: IAR + RelationalAnalysis = 0.62 + 1.65 = 2.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -809.0066789, upper bound: 809.0066789

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4691278, upper bound: 808.5062695
time: 0.62 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0038423, upper bound: 809.0038429
time: 0.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.27 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 4, lower bound: -806.4691278, upper bound: 808.5062695
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 4, lower bound: -809.0038423, upper bound: 809.0038429

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -355.7917175, 298.0696411, -461.5992737, 385.9967346, -741.7883301, 759.6687012
1: -284.4114380, 290.6557922, -370.1784058, 374.3149719, -658.7263794, 660.8341675
2: -411.3229675, 317.9904480, -538.0414429, 408.4810486, -819.8040161, 856.0318604
3: -162.9068451, 405.6134033, -208.3363800, 526.5999756, -689.5068359, 613.9497070
4: -458.9186401, 313.6753845, -599.3458252, 404.4221802, -863.3408203, 913.0212402

Time for backsubstitution: 0.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.4684386, upper bound: 806.4684386
time: 0.57 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4684386, upper bound: 808.5062695
time: 0.60 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -455.0569763, 380.6590271, -465.2375793, 388.7125854, -843.7695312, 845.8966064
1: -364.9006348, 369.0178833, -373.0900269, 376.9524841, -741.8531494, 742.1078491
2: -530.4962158, 402.6548462, -542.3095093, 411.3765564, -941.8726807, 944.9643555
3: -205.3341675, 519.1251221, -209.7973175, 530.6040649, -735.9381104, 728.9224243
4: -590.9782715, 398.9001770, -604.0916138, 407.2537537, -998.2319336, 1002.9916992

Time for backsubstitution: 0.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5062695, upper bound: 806.4691278
time: 0.57 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5062695, upper bound: 806.4691278
time: 0.48 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.64 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 1.64
Output dim: 4, lower bound: -806.4684386, upper bound: 806.4684386
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.64
Output dim: 4, lower bound: -806.4684386, upper bound: 808.5062695
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.64
Output dim: 4, lower bound: -808.5062695, upper bound: 806.4691278
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.64
Output dim: 4, lower bound: -808.5062695, upper bound: 806.4691278

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -355.7917175, 298.0696411, -455.0569763, 380.6590271, -736.4506836, 753.1265869
1: -284.4114380, 290.6557922, -364.9006348, 369.0178833, -653.4293213, 655.5563965
2: -411.3229675, 317.9904480, -530.4962158, 402.6548462, -813.9777832, 848.4865723
3: -162.9068451, 405.6134033, -205.3341675, 519.1251221, -682.0319824, 610.9474487
4: -458.9186401, 313.6753845, -590.9782715, 398.9001770, -857.8187256, 904.6536865

Time for backsubstitution: 0.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8374152, upper bound: 807.9988380
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2175174, upper bound: 806.2175181
time: 0.58 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -455.0569763, 380.6590271, -355.7917175, 298.0696411, -753.1265869, 736.4506836
1: -364.9006348, 369.0178833, -284.4114380, 290.6557922, -655.5563965, 653.4293213
2: -530.4962158, 402.6548462, -411.3229675, 317.9904480, -848.4865723, 813.9777832
3: -205.3341675, 519.1251221, -162.9068451, 405.6134033, -610.9473877, 682.0319824
4: -590.9782715, 398.9001770, -458.9186401, 313.6753845, -904.6536865, 857.8186646

Time for backsubstitution: 0.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4604153, upper bound: 806.4636033
time: 0.53 seconds

## Relational analysis of NS_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9988370, upper bound: 805.8381632
time: 0.68 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4488128, upper bound: 806.2184499
time: 0.53 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -455.0569763, 380.6590271, -455.0569763, 380.6590271, -835.7160034, 835.7160034
1: -364.9006348, 369.0178833, -364.9006348, 369.0178833, -733.9185181, 733.9185181
2: -530.4962158, 402.6548462, -530.4962158, 402.6548462, -933.1510620, 933.1510620
3: -205.3341675, 519.1251221, -205.3341675, 519.1251221, -724.4591675, 724.4591675
4: -590.9782715, 398.9001770, -590.9782715, 398.9001770, -989.8784180, 989.8782959

Time for backsubstitution: 0.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4604163, upper bound: 808.8222452
time: 0.51 seconds

## Relational analysis of NS_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5062003, upper bound: 808.5813317
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3347836, upper bound: 808.5812559
time: 0.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.42 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 4, lower bound: -805.8374152, upper bound: 807.9988380
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.42
Output dim: 4, lower bound: -806.2175174, upper bound: 806.2175181
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 4, lower bound: -807.9988370, upper bound: 805.8381632
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 4, lower bound: -808.4488128, upper bound: 806.2184499
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 4, lower bound: -808.5062003, upper bound: 808.5813317
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 4, lower bound: -808.3347836, upper bound: 808.5812559

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -251.5948334, 219.6874542, -438.6151123, 368.5163879, -620.1110840, 658.3025513
1: -200.1122742, 214.7951965, -351.6083069, 357.1553955, -557.2677002, 566.4035034
2: -288.3356018, 235.8116150, -511.1228027, 389.7881470, -678.1235962, 746.9343872
3: -120.3001709, 288.3111572, -198.6533203, 500.6182251, -620.9183960, 486.9644165
4: -322.8778687, 232.7944031, -569.5524292, 386.3522339, -709.2301025, 802.3467407

Time for backsubstitution: 0.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.9124391, upper bound: 806.7042999
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8372645, upper bound: 807.9988370
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -438.6151123, 368.5163879, -251.5948334, 219.6874542, -658.3025513, 620.1110840
1: -351.6083069, 357.1553955, -200.1122742, 214.7951965, -566.4035034, 557.2677002
2: -511.1228027, 389.7881470, -288.3356018, 235.8116150, -746.9344482, 678.1235962
3: -198.6533203, 500.6182251, -120.3001709, 288.3111572, -486.9644165, 620.9183960
4: -569.5524292, 386.3522339, -322.8778687, 232.7944031, -802.3467407, 709.2301025

Time for backsubstitution: 0.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7042999, upper bound: 804.9124391
time: 0.55 seconds

## Relational analysis of NS_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9988370, upper bound: 805.8379954
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -455.0569763, 380.6590271, -348.0386353, 292.4441223, -747.5010986, 728.6975708
1: -364.9006348, 369.0178833, -278.1151123, 285.1520081, -650.0526123, 647.1329956
2: -530.4962158, 402.6548462, -402.1842957, 312.0704346, -842.5665894, 804.8391113
3: -205.3341675, 519.1251221, -159.7416534, 396.7953796, -602.1293945, 678.8667603
4: -590.9782715, 398.9001770, -448.7807617, 307.8369446, -898.8151855, 847.6809082

Time for backsubstitution: 0.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8261256, upper bound: 806.2181072
time: 0.53 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1885050, upper bound: 806.2181072
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -437.7723999, 367.4412842, -453.9830627, 379.6953125, -817.4675293, 821.4243164
1: -350.6974182, 356.2439270, -364.0279846, 368.1007385, -718.7981567, 720.2719116
2: -509.7792053, 388.8671265, -529.2156982, 401.6600037, -911.4390869, 918.0828247
3: -198.0632477, 499.4380493, -204.8486328, 517.8768311, -715.9400635, 704.2866821
4: -568.3402710, 385.4368286, -589.5609741, 397.9028931, -966.2431641, 974.9977417

Time for backsubstitution: 0.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5811934
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5811934
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -444.8894653, 372.2908936, -454.6670532, 380.3390503, -825.2284546, 826.9579468
1: -356.6080017, 360.9292603, -364.5827637, 368.7083130, -725.3162842, 725.5120239
2: -518.1353760, 393.8309326, -530.0238647, 402.3171387, -920.4525146, 923.8547974
3: -200.9437408, 507.3840332, -205.1652374, 518.6757812, -719.6195068, 712.5491333
4: -577.2812500, 390.2192383, -590.4550171, 398.5679626, -975.8492432, 980.6741943

Time for backsubstitution: 0.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5812554
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5812554
time: 0.60 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.82 seconds
NS_A1_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 1.82
Output dim: 4, lower bound: -804.9124391, upper bound: 806.7042999
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 1.82
Output dim: 4, lower bound: -805.8372645, upper bound: 807.9988370
NS_A2_B1_B1_B1, status: Status.VERIFIED, split count: 4, time: 1.82
Output dim: 4, lower bound: -806.7042999, upper bound: 804.9124391
NS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 1.82
Output dim: 4, lower bound: -807.9988370, upper bound: 805.8379954
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 1.82
Output dim: 4, lower bound: -807.8261256, upper bound: 806.2181072
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 1.82
Output dim: 4, lower bound: -808.1885050, upper bound: 806.2181072
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.82
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5811934
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.82
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5811934
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.82
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5812554
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.82
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5812554

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -243.7677917, 213.4292908, -438.2617493, 368.2253418, -611.9930420, 651.6910400
1: -193.7469482, 208.7287445, -351.3199463, 356.8738708, -550.6207886, 560.0486450
2: -278.9052429, 229.1844482, -510.6934204, 389.4810181, -668.3860474, 739.8778687
3: -116.8389282, 279.3439636, -198.5010071, 500.2093201, -617.0482178, 477.8448792
4: -312.4699707, 226.3032074, -569.0766602, 386.0499878, -698.5199585, 795.3798218

Time for backsubstitution: 0.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.8369747, upper bound: 806.2218813
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8369747, upper bound: 807.9988370
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -438.2617493, 368.2253418, -243.7677917, 213.4292908, -651.6910400, 611.9930420
1: -351.3199463, 356.8738708, -193.7469482, 208.7287445, -560.0486450, 550.6207886
2: -510.6934204, 389.4810181, -278.9052429, 229.1844482, -739.8778687, 668.3861084
3: -198.5010071, 500.2093201, -116.8389282, 279.3439636, -477.8448792, 617.0482178
4: -569.0766602, 386.0499878, -312.4699707, 226.3032074, -795.3798218, 698.5199585

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2218813, upper bound: 805.8369747
time: 0.58 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2218813, upper bound: 805.8379954
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -453.9830627, 379.6953125, -332.7971191, 281.7926941, -735.7756958, 712.4923706
1: -364.0279846, 368.1007385, -265.7110901, 274.8519287, -638.8798218, 633.8118286
2: -529.2156982, 401.6600037, -384.1240845, 300.9845886, -830.2003174, 785.7838745
3: -204.8486328, 517.8768311, -153.6954041, 379.7464600, -584.5950928, 671.5722656
4: -589.5609741, 397.9028931, -429.0441895, 297.0279236, -886.5888672, 826.9470215

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8261256, upper bound: 806.2181072
time: 0.69 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8261256, upper bound: 806.2181072
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -454.6670532, 380.3390503, -339.1815186, 285.2177429, -739.8847656, 719.5205688
1: -364.5827637, 368.7083130, -270.9281616, 278.1785583, -642.7613525, 639.6364746
2: -530.0238647, 402.3171387, -391.5458069, 304.4880066, -834.5118408, 793.8629150
3: -205.1652374, 518.6757812, -155.5240631, 386.5900574, -591.7549438, 674.1998291
4: -590.4550171, 398.5679626, -436.9592896, 300.3887939, -890.8438110, 835.5272217

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1885050, upper bound: 806.2181072
time: 0.55 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1885050, upper bound: 806.2181072
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -437.7723999, 367.4412842, -437.7723999, 367.4412842, -805.2136230, 805.2136230
1: -350.6974182, 356.2439270, -350.6974182, 356.2439270, -706.9413452, 706.9413452
2: -509.7792053, 388.8671265, -509.7792053, 388.8671265, -898.6461792, 898.6463013
3: -198.0632477, 499.4380493, -198.0632477, 499.4380493, -697.5012817, 697.5012817
4: -568.3402710, 385.4368286, -568.3402710, 385.4368286, -953.7770996, 953.7770996

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9355054, upper bound: 808.3422381
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9894669, upper bound: 808.3425082
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -437.7723999, 367.4412842, -444.8894653, 372.2908936, -810.0632935, 812.3307495
1: -350.6974182, 356.2439270, -356.6080017, 360.9292603, -711.6267090, 712.8518677
2: -509.7792053, 388.8671265, -518.1353760, 393.8309326, -903.6101074, 907.0024414
3: -198.0632477, 499.4380493, -200.9437408, 507.3840332, -705.4472656, 700.3817139
4: -568.3402710, 385.4368286, -577.2812500, 390.2192383, -958.5594482, 962.7180786

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9892188, upper bound: 808.5467403
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9894669, upper bound: 808.3425082
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -444.8894653, 372.2908936, -437.7723999, 367.4412842, -812.3307495, 810.0632324
1: -356.6080017, 360.9292603, -350.6974182, 356.2439270, -712.8519287, 711.6267090
2: -518.1353760, 393.8309326, -509.7792053, 388.8671265, -907.0024414, 903.6101074
3: -200.9437408, 507.3840332, -198.0632477, 499.4380493, -700.3817749, 705.4472656
4: -577.2812500, 390.2192383, -568.3402710, 385.4368286, -962.7180786, 958.5593872

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5470841, upper bound: 808.3424601
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3424873, upper bound: 808.3422641
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -444.8894653, 372.2908936, -444.8894653, 372.2908936, -817.1803589, 817.1803589
1: -356.6080017, 360.9292603, -356.6080017, 360.9292603, -717.5372314, 717.5372314
2: -518.1353760, 393.8309326, -518.1353760, 393.8309326, -911.9663086, 911.9663086
3: -200.9437408, 507.3840332, -200.9437408, 507.3840332, -708.3276978, 708.3277588
4: -577.2812500, 390.2192383, -577.2812500, 390.2192383, -967.5004883, 967.5004883

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2226965, upper bound: 807.4891025
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5199483, upper bound: 808.5198006
time: 0.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 1.96 seconds
NS_A1_B2_A1_A2_B1, status: Status.VERIFIED, split count: 5, time: 1.96
Output dim: 4, lower bound: -805.8369747, upper bound: 806.2218813
NS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 1.96
Output dim: 4, lower bound: -805.8369747, upper bound: 807.9988370
NS_A2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 1.96
Output dim: 4, lower bound: -806.2218813, upper bound: 805.8369747
NS_A2_B1_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 1.96
Output dim: 4, lower bound: -806.2218813, upper bound: 805.8379954
NS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.96
Output dim: 4, lower bound: -807.8261256, upper bound: 806.2181072
NS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.96
Output dim: 4, lower bound: -807.8261256, upper bound: 806.2181072
NS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.96
Output dim: 4, lower bound: -808.1885050, upper bound: 806.2181072
NS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.96
Output dim: 4, lower bound: -808.1885050, upper bound: 806.2181072
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.96
Output dim: 4, lower bound: -808.9355054, upper bound: 808.3422381
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.96
Output dim: 4, lower bound: -808.9894669, upper bound: 808.3425082
NS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 1.96
Output dim: 4, lower bound: -808.9892188, upper bound: 808.5467403
NS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 1.96
Output dim: 4, lower bound: -808.9894669, upper bound: 808.3425082
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.96
Output dim: 4, lower bound: -808.5470841, upper bound: 808.3424601
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.96
Output dim: 4, lower bound: -808.3424873, upper bound: 808.3422641
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.96
Output dim: 4, lower bound: -806.2226965, upper bound: 807.4891025
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.96
Output dim: 4, lower bound: -808.5199483, upper bound: 808.5198006

## BFS NS instance: NS_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -243.7677917, 213.4292908, -429.2519836, 359.3202209, -603.0878906, 642.6811523
1: -193.7469482, 208.7287445, -343.8303223, 348.5745239, -542.3214111, 552.5589600
2: -278.9052429, 229.1844482, -499.4364319, 380.9186707, -659.8236694, 728.6207275
3: -116.8389282, 279.3439636, -194.6154480, 489.0296326, -605.8684692, 473.9593811
4: -312.4699707, 226.3032074, -556.5651245, 376.9931335, -689.4631348, 782.8682251

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5665284, upper bound: 807.7844364
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8367768, upper bound: 807.9988028
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8369254, upper bound: 807.9959642
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -437.7723999, 367.4412842, -332.7971191, 281.7926941, -719.5649414, 700.2384033
1: -350.6974182, 356.2439270, -265.7110901, 274.8519287, -625.5492554, 621.9550171
2: -509.7792053, 388.8671265, -384.1240845, 300.9845886, -810.7637329, 772.9910889
3: -198.0632477, 499.4380493, -153.6954041, 379.7464600, -577.8096924, 653.1334229
4: -568.3402710, 385.4368286, -429.0441895, 297.0279236, -865.3681641, 814.4809570

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B2_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.5346103, upper bound: 805.0733274
time: 0.59 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8189308, upper bound: 806.2180983
time: 0.51 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -444.8894653, 372.2908936, -332.7971191, 281.7926941, -726.6820068, 705.0880127
1: -356.6080017, 360.9292603, -265.7110901, 274.8519287, -631.4597778, 626.6403809
2: -518.1353760, 393.8309326, -384.1240845, 300.9845886, -819.1199951, 777.9550171
3: -200.9437408, 507.3840332, -153.6954041, 379.7464600, -580.6901855, 661.0794678
4: -577.2812500, 390.2192383, -429.0441895, 297.0279236, -874.3092041, 819.2633057

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B2_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.5346103, upper bound: 805.0733274
time: 0.50 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8189308, upper bound: 806.2180983
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -437.7723999, 367.4412842, -339.1815186, 285.2177429, -722.9901123, 706.6228027
1: -350.6974182, 356.2439270, -270.9281616, 278.1785583, -628.8759766, 627.1720581
2: -509.7792053, 388.8671265, -391.5458069, 304.4880066, -814.2671509, 780.4129028
3: -198.0632477, 499.4380493, -155.5240631, 386.5900574, -584.6531372, 654.9620361
4: -568.3402710, 385.4368286, -436.9592896, 300.3887939, -868.7290649, 822.3961182

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7862252, upper bound: 805.9774198
time: 0.57 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -444.8894653, 372.2908936, -339.1815186, 285.2177429, -730.1071777, 711.4724121
1: -356.6080017, 360.9292603, -270.9281616, 278.1785583, -634.7865601, 631.8574219
2: -518.1353760, 393.8309326, -391.5458069, 304.4880066, -822.6232910, 785.3767090
3: -200.9437408, 507.3840332, -155.5240631, 386.5900574, -587.5335693, 662.9080200
4: -577.2812500, 390.2192383, -436.9592896, 300.3887939, -877.6700439, 827.1785278

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7862252, upper bound: 805.9774198
time: 0.54 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8482231, upper bound: 803.5246119
time: 0.58 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -362.6666565, 305.9226074, -418.7742615, 351.8649597, -714.5314941, 724.6968994
1: -290.2025757, 296.8375549, -335.3200378, 341.1825867, -631.3851318, 632.1575317
2: -421.4447021, 324.4236450, -487.3020630, 372.5691223, -794.0137939, 811.7256470
3: -164.7423096, 413.9035950, -189.6507568, 477.7037048, -642.4460449, 603.5543213
4: -469.5854187, 321.2560120, -543.3257446, 369.2556458, -838.8410645, 864.5817261

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9352549, upper bound: 808.9346287
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9352549, upper bound: 808.9889079
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -428.7608032, 359.9247742, -437.7723999, 367.4412842, -796.2020264, 797.6970825
1: -343.3657837, 348.9992065, -350.6974182, 356.2439270, -699.6097412, 699.6966553
2: -499.1161499, 381.0668945, -509.7792053, 388.8671265, -887.9830933, 890.8460693
3: -194.1269379, 489.1353455, -198.0632477, 499.4380493, -693.5650024, 687.1984863
4: -556.5690918, 377.6802368, -568.3402710, 385.4368286, -942.0059204, 946.0205078

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9892188, upper bound: 808.9346287
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9892188, upper bound: 808.9890459
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -418.7742615, 351.8649597, -380.1496887, 318.2424622, -737.0167236, 732.0145874
1: -335.3200378, 341.1825867, -304.5241394, 308.7597961, -644.0798340, 645.7067261
2: -487.3020630, 372.5691223, -441.9195862, 337.3076477, -824.6095581, 814.4887085
3: -189.6507568, 477.7037048, -171.6442566, 433.2800598, -622.9307861, 649.3478394
4: -543.3257446, 369.2556458, -491.8168945, 333.9521179, -877.2778320, 861.0725098

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9354571, upper bound: 808.3422381
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9354571, upper bound: 808.3425082
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -437.7723999, 367.4412842, -436.6898499, 365.1800842, -802.9523926, 804.1311035
1: -350.6974182, 356.2439270, -349.9454956, 354.0716248, -704.7690430, 706.1893921
2: -509.7792053, 388.8671265, -508.4296265, 386.5229187, -896.3020020, 897.2967529
3: -198.0632477, 499.4380493, -197.2324371, 497.9155884, -695.9788208, 696.6704102
4: -568.3402710, 385.4368286, -566.5614014, 382.8739319, -951.2141113, 951.9982300

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.1571634, upper bound: 806.5160954
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9552012, upper bound: 808.3363816
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -380.1496887, 318.2424622, -418.7742615, 351.8649597, -732.0145874, 737.0167236
1: -304.5241394, 308.7597961, -335.3200378, 341.1825867, -645.7067261, 644.0798340
2: -441.9195862, 337.3076477, -487.3020630, 372.5691223, -814.4887085, 824.6095581
3: -171.6442566, 433.2800598, -189.6507568, 477.7037048, -649.3478394, 622.9307861
4: -491.8168945, 333.9521179, -543.3257446, 369.2556458, -861.0725098, 877.2778320

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3415659, upper bound: 808.3726854
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3415659, upper bound: 808.5543601
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -436.6898499, 365.1800842, -437.7723999, 367.4412842, -804.1311035, 802.9523926
1: -349.9454956, 354.0716248, -350.6974182, 356.2439270, -706.1893921, 704.7690430
2: -508.4296265, 386.5229187, -509.7792053, 388.8671265, -897.2967529, 896.3020020
3: -197.2324371, 497.9155884, -198.0632477, 499.4380493, -696.6704102, 695.9788208
4: -566.5614014, 382.8739319, -568.3402710, 385.4368286, -951.9982300, 951.2140503

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8767470, upper bound: 805.3810704
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3364670, upper bound: 808.5378525
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -351.7501831, 302.6012573, -428.8754883, 360.4992065, -712.2493896, 731.4767456
1: -281.1611328, 292.9727173, -343.6613159, 349.3997192, -630.5607300, 636.6340332
2: -408.2001648, 320.2536926, -499.2624817, 381.3300781, -789.5302734, 819.5161133
3: -162.5398102, 402.2868347, -194.4609222, 489.3500671, -651.8898926, 596.7477417
4: -455.7276001, 318.0516357, -556.4102783, 378.0300903, -833.7576904, 874.4619141

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2220321, upper bound: 806.2217430
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2220321, upper bound: 807.4890994
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -419.5442810, 351.3311157, -444.8894653, 372.2908936, -791.8352051, 796.2205811
1: -335.8997192, 340.8306885, -356.6080017, 360.9292603, -696.8289795, 697.4386597
2: -487.5963440, 372.4834900, -518.1353760, 393.8309326, -881.4272461, 890.6188965
3: -190.3561554, 477.7799683, -200.9437408, 507.3840332, -697.7401123, 678.7236938
4: -543.4680786, 368.7157288, -577.2812500, 390.2192383, -933.6873169, 945.9969482

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.2189456, upper bound: 806.2218399
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.2189551, upper bound: 808.5198006
time: 0.54 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 1.79 seconds
NS_A1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 1.79
Output dim: 4, lower bound: -805.8367768, upper bound: 807.9988028
NS_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 1.79
Output dim: 4, lower bound: -805.8369254, upper bound: 807.9959642
NS_A2_B1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.79
Output dim: 4, lower bound: -806.5346103, upper bound: 805.0733274
NS_A2_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.79
Output dim: 4, lower bound: -807.8189308, upper bound: 806.2180983
NS_A2_B1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 1.79
Output dim: 4, lower bound: -806.5346103, upper bound: 805.0733274
NS_A2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.79
Output dim: 4, lower bound: -807.8189308, upper bound: 806.2180983
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.79
Output dim: 4, lower bound: -808.9352549, upper bound: 808.9346287
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.79
Output dim: 4, lower bound: -808.9352549, upper bound: 808.9889079
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.79
Output dim: 4, lower bound: -808.9892188, upper bound: 808.9346287
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.79
Output dim: 4, lower bound: -808.9892188, upper bound: 808.9890459
NS_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.79
Output dim: 4, lower bound: -808.9354571, upper bound: 808.3422381
NS_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.79
Output dim: 4, lower bound: -808.9354571, upper bound: 808.3425082
NS_A2_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 1.79
Output dim: 4, lower bound: -805.1571634, upper bound: 806.5160954
NS_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.79
Output dim: 4, lower bound: -808.9552012, upper bound: 808.3363816
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.79
Output dim: 4, lower bound: -808.3415659, upper bound: 808.3726854
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.79
Output dim: 4, lower bound: -808.3415659, upper bound: 808.5543601
NS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 1.79
Output dim: 4, lower bound: -806.8767470, upper bound: 805.3810704
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.79
Output dim: 4, lower bound: -808.3364670, upper bound: 808.5378525
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.79
Output dim: 4, lower bound: -806.2220321, upper bound: 806.2217430
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.79
Output dim: 4, lower bound: -806.2220321, upper bound: 807.4890994
NS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 1.79
Output dim: 4, lower bound: -807.2189456, upper bound: 806.2218399
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.79
Output dim: 4, lower bound: -807.2189551, upper bound: 808.5198006

## BFS NS instance: NS_A1_B2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -230.1211853, 202.5869751, -363.5061035, 304.4846802, -534.6057739, 566.0930786
1: -182.7263794, 198.2754822, -290.9722595, 295.5873718, -478.3137512, 489.2476501
2: -262.9144897, 217.8549194, -422.1159058, 323.2605896, -586.1749878, 639.9708252
3: -110.7996674, 263.9035034, -164.7344055, 413.8491211, -524.6487427, 428.6379089
4: -294.6611938, 215.1001282, -469.9195251, 319.7472839, -614.4084473, 685.0196533

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.8062471, upper bound: 805.9442572
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.3322637, upper bound: 807.6707709
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5943819, upper bound: 807.9909316
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -243.7677917, 213.4292908, -421.9108276, 352.9677429, -596.7355347, 635.3400879
1: -193.7469482, 208.7287445, -337.8547058, 342.4287415, -536.1755981, 546.5833740
2: -278.9052429, 229.1844482, -490.7287292, 374.3025513, -653.2075806, 719.9132080
3: -116.8389282, 279.3439636, -191.2692871, 480.5101318, -597.3489990, 470.6131287
4: -312.4699707, 226.3032074, -546.9573364, 370.4477539, -682.9177246, 773.2604980

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8378312, upper bound: 807.9472276
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8378312, upper bound: 807.9959610
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -437.7723999, 367.4412842, -330.0129395, 279.4838257, -717.2560425, 697.4542236
1: -350.6974182, 356.2439270, -263.4294434, 272.6274414, -623.3248291, 619.6733398
2: -509.7792053, 388.8671265, -380.7520752, 298.5825806, -808.3616943, 769.6192017
3: -198.0632477, 499.4380493, -152.3882141, 376.5030823, -574.5663452, 651.8262329
4: -568.3402710, 385.4368286, -425.3255005, 294.6168213, -862.9569702, 810.7622681

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3212714, upper bound: 805.9776549
time: 0.49 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.3667908, upper bound: 804.0942061
time: 0.67 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.8784994, upper bound: 806.2165139
time: 0.57 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.8784994, upper bound: 806.2183171
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -444.8894653, 372.2908936, -330.0129395, 279.4838257, -724.3732300, 702.3038330
1: -356.6080017, 360.9292603, -263.4294434, 272.6274414, -629.2353516, 624.3587036
2: -518.1353760, 393.8309326, -380.7520752, 298.5825806, -816.7179565, 774.5830078
3: -200.9437408, 507.3840332, -152.3882141, 376.5030823, -577.4468384, 659.7722168
4: -577.2812500, 390.2192383, -425.3255005, 294.6168213, -871.8980713, 815.5446777

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5424493, upper bound: 806.2170170
time: 0.55 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8189308, upper bound: 806.2174581
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -362.6666565, 305.9226074, -362.6666565, 305.9226074, -668.5892334, 668.5892334
1: -290.2025757, 296.8375549, -290.2025757, 296.8375549, -587.0401611, 587.0401611
2: -421.4447021, 324.4236450, -421.4447021, 324.4236450, -745.8683472, 745.8683472
3: -164.7423096, 413.9035950, -164.7423096, 413.9035950, -578.6458740, 578.6458740
4: -469.5854187, 321.2560120, -469.5854187, 321.2560120, -790.8414307, 790.8414307

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9097628, upper bound: 806.6795771
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7124686, upper bound: 806.6796127
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -362.6666565, 305.9226074, -428.7608032, 359.9247742, -722.5913696, 734.6833496
1: -290.2025757, 296.8375549, -343.3657837, 348.9992065, -639.2017822, 640.2033081
2: -421.4447021, 324.4236450, -499.1161499, 381.0668945, -802.5115967, 823.5396729
3: -164.7423096, 413.9035950, -194.1269379, 489.1353455, -653.8776245, 608.0305176
4: -469.5854187, 321.2560120, -556.5690918, 377.6802368, -847.2656250, 877.8250732

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9097628, upper bound: 806.6797948
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7124686, upper bound: 806.6798286
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -428.7608032, 359.9247742, -362.6666565, 305.9226074, -734.6834106, 722.5913696
1: -343.3657837, 348.9992065, -290.2025757, 296.8375549, -640.2033081, 639.2017822
2: -499.1161499, 381.0668945, -421.4447021, 324.4236450, -823.5396729, 802.5115967
3: -194.1269379, 489.1353455, -164.7423096, 413.9035950, -608.0305176, 653.8776245
4: -556.5690918, 377.6802368, -469.5854187, 321.2560120, -877.8250732, 847.2656250

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.6802539, upper bound: 808.9086807
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.6798495, upper bound: 806.6795573
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -428.7608032, 359.9247742, -428.7608032, 359.9247742, -788.6854858, 788.6854858
1: -343.3657837, 348.9992065, -343.3657837, 348.9992065, -692.3649902, 692.3649902
2: -499.1161499, 381.0668945, -499.1161499, 381.0668945, -880.1829224, 880.1829224
3: -194.1269379, 489.1353455, -194.1269379, 489.1353455, -683.2622681, 683.2622681
4: -556.5690918, 377.6802368, -556.5690918, 377.6802368, -934.2493286, 934.2493286

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.6802539, upper bound: 808.9135652
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.6798495, upper bound: 806.6797822
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -362.6666565, 305.9226074, -380.1496887, 318.2424622, -680.9091187, 686.0722656
1: -290.2025757, 296.8375549, -304.5241394, 308.7597961, -598.9624023, 601.3616943
2: -421.4447021, 324.4236450, -441.9195862, 337.3076477, -758.7522583, 766.3432617
3: -164.7423096, 413.9035950, -171.6442566, 433.2800598, -598.0223389, 585.5477295
4: -469.5854187, 321.2560120, -491.8168945, 333.9521179, -803.5375366, 813.0728760

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8686211, upper bound: 808.3607526
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8685352, upper bound: 808.3289309
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -428.7608032, 359.9247742, -380.1496887, 318.2424622, -747.0031738, 740.0744019
1: -343.3657837, 348.9992065, -304.5241394, 308.7597961, -652.1256104, 653.5233154
2: -499.1161499, 381.0668945, -441.9195862, 337.3076477, -836.4235840, 822.9864502
3: -194.1269379, 489.1353455, -171.6442566, 433.2800598, -627.4069824, 660.7793579
4: -556.5690918, 377.6802368, -491.8168945, 333.9521179, -890.5212402, 869.4971313

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.6802485, upper bound: 808.5214522
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -434.0058594, 364.2015076, -436.6898499, 365.1800842, -799.1859131, 800.8913574
1: -347.6188049, 353.1040955, -349.9454956, 354.0716248, -701.6904297, 703.0494995
2: -505.2263184, 385.4602966, -508.4296265, 386.5229187, -891.7492676, 893.8898926
3: -196.3519897, 495.0065613, -197.2324371, 497.9155884, -694.2675781, 692.2387695
4: -563.3115845, 382.0725098, -566.5614014, 382.8739319, -946.1854248, 948.6339111

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8957769, upper bound: 808.3361203
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8957769, upper bound: 808.3363816
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -380.1496887, 318.2424622, -362.6666565, 305.9226074, -686.0722656, 680.9091187
1: -304.5241394, 308.7597961, -290.2025757, 296.8375549, -601.3616943, 598.9624023
2: -441.9195862, 337.3076477, -421.4447021, 324.4236450, -766.3432617, 758.7522583
3: -171.6442566, 433.2800598, -164.7423096, 413.9035950, -585.5477295, 598.0223389
4: -491.8168945, 333.9521179, -469.5854187, 321.2560120, -813.0728760, 803.5375366

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.7297966, upper bound: 808.9050506
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7073379, upper bound: 806.6747701
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -380.1496887, 318.2424622, -428.7608032, 359.9247742, -740.0744019, 747.0031738
1: -304.5241394, 308.7597961, -343.3657837, 348.9992065, -653.5233154, 652.1256104
2: -441.9195862, 337.3076477, -499.1161499, 381.0668945, -822.9864502, 836.4235840
3: -171.6442566, 433.2800598, -194.1269379, 489.1353455, -660.7793579, 627.4069824
4: -491.8168945, 333.9521179, -556.5690918, 377.6802368, -869.4971313, 890.5212402

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3925604, upper bound: 806.6746214
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7073379, upper bound: 806.6751797
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -436.6898499, 365.1800842, -434.0058594, 364.2015076, -800.8913574, 799.1859131
1: -349.9454956, 354.0716248, -347.6188049, 353.1040955, -703.0495605, 701.6904297
2: -508.4296265, 386.5229187, -505.2263184, 385.4602966, -893.8898926, 891.7492676
3: -197.2324371, 497.9155884, -196.3519897, 495.0065613, -692.2388306, 694.2675781
4: -566.5614014, 382.8739319, -563.3115845, 382.0725098, -948.6339111, 946.1854248

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3354437, upper bound: 808.3665388
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3354433, upper bound: 808.3665388
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -351.7501831, 302.6012573, -419.5442810, 351.3311157, -703.0812988, 722.1455078
1: -281.1611328, 292.9727173, -335.8997192, 340.8306885, -621.9916382, 628.8724365
2: -408.2001648, 320.2536926, -487.5963440, 372.4834900, -780.6836548, 807.8500366
3: -162.5398102, 402.2868347, -190.3561554, 477.7799683, -640.3197632, 592.6429443
4: -455.7276001, 318.0516357, -543.4680786, 368.7157288, -824.4432983, 861.5197144

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0270021, upper bound: 806.0962950
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0272371, upper bound: 806.6646082
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -419.5442810, 351.3311157, -419.5442810, 351.3311157, -770.8753662, 770.8753662
1: -335.8997192, 340.8306885, -335.8997192, 340.8306885, -676.7304077, 676.7304077
2: -487.5963440, 372.4834900, -487.5963440, 372.4834900, -860.0797729, 860.0798340
3: -190.3561554, 477.7799683, -190.3561554, 477.7799683, -668.1361084, 668.1361084
4: -543.4680786, 368.7157288, -543.4680786, 368.7157288, -912.1837769, 912.1838379

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.2189318, upper bound: 808.2030398
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.5986834, upper bound: 808.2024878
time: 0.64 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.01 seconds
NS_A1_B2_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -805.3322637, upper bound: 807.6707709
NS_A1_B2_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -805.5943819, upper bound: 807.9909316
NS_A1_B2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -805.8378312, upper bound: 807.9472276
NS_A1_B2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -805.8378312, upper bound: 807.9959610
NS_A2_B1_B2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 4, lower bound: -805.8784994, upper bound: 806.2165139
NS_A2_B1_B2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 4, lower bound: -805.8784994, upper bound: 806.2183171
NS_A2_B1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -807.5424493, upper bound: 806.2170170
NS_A2_B1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -807.8189308, upper bound: 806.2174581
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -808.9097628, upper bound: 806.6795771
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 4, lower bound: -806.7124686, upper bound: 806.6796127
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -808.9097628, upper bound: 806.6797948
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 4, lower bound: -806.7124686, upper bound: 806.6798286
NS_A2_B2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -806.6802539, upper bound: 808.9086807
NS_A2_B2_A1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 4, lower bound: -806.6798495, upper bound: 806.6795573
NS_A2_B2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -806.6802539, upper bound: 808.9135652
NS_A2_B2_A1_B1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 4, lower bound: -806.6798495, upper bound: 806.6797822
NS_A2_B2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -808.8686211, upper bound: 808.3607526
NS_A2_B2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -808.8685352, upper bound: 808.3289309
NS_A2_B2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -808.8957769, upper bound: 808.3361203
NS_A2_B2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -808.8957769, upper bound: 808.3363816
NS_A2_B2_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -806.7297966, upper bound: 808.9050506
NS_A2_B2_A2_B1_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 4, lower bound: -806.7073379, upper bound: 806.6747701
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -808.3925604, upper bound: 806.6746214
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 4, lower bound: -806.7073379, upper bound: 806.6751797
NS_A2_B2_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -808.3354437, upper bound: 808.3665388
NS_A2_B2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -808.3354433, upper bound: 808.3665388
NS_A2_B2_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 4, lower bound: -806.0270021, upper bound: 806.0962950
NS_A2_B2_A2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 4, lower bound: -806.0272371, upper bound: 806.6646082
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -807.2189318, upper bound: 808.2030398
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 4, lower bound: -806.5986834, upper bound: 808.2024878

## BFS NS instance: NS_A1_B2_A1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -202.6057129, 181.7199097, -357.0698853, 299.7659607, -502.3716736, 538.7897949
1: -160.6386108, 177.8678741, -285.8115234, 290.9804077, -451.6190186, 463.6793518
2: -230.7832336, 195.4892883, -414.5695801, 318.2037964, -548.9870605, 610.0588379
3: -99.5082397, 232.7431793, -162.1309967, 406.8362122, -506.3444519, 394.8741455
4: -259.0898743, 193.3060455, -461.6263428, 314.8148804, -573.9047852, 654.9322510

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.4478871, upper bound: 806.8783384
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -804.9129701, upper bound: 807.5049396
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -301.3088684, 257.3271179, -359.9024353, 301.8214111, -603.1302490, 617.2295532
1: -240.1283112, 250.6216888, -288.0487671, 292.9872437, -533.1155396, 538.6704712
2: -345.9265137, 274.6401062, -417.8614807, 320.4252625, -666.3518066, 692.5014648
3: -142.0344543, 343.3493958, -163.2777863, 409.8174438, -551.8519287, 506.6271057
4: -386.8622437, 271.9160156, -465.2668457, 316.9648132, -703.8270264, 737.1828613

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5665718, upper bound: 807.6211005
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5665719, upper bound: 807.9909316
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -243.7677917, 213.4292908, -402.4073181, 338.4165344, -582.1842651, 615.8364868
1: -193.7469482, 208.7287445, -321.8751526, 328.3661804, -522.1130981, 530.6038208
2: -278.9052429, 229.1844482, -467.4528809, 359.1240234, -638.0291748, 696.6372681
3: -116.8389282, 279.3439636, -183.3221741, 458.4645691, -575.3034668, 462.6660767
4: -312.4699707, 226.3032074, -521.5133667, 355.6081543, -668.0780640, 747.8165283

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.3323501, upper bound: 807.9170288
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_A2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5942547, upper bound: 807.8635990
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -243.7677917, 213.4292908, -411.4728699, 344.4328003, -588.2005615, 624.9021606
1: -193.7469482, 208.7287445, -329.3374329, 334.1509399, -527.8978882, 538.0661621
2: -278.9052429, 229.1844482, -478.0291443, 365.2938538, -644.1990967, 707.2136230
3: -116.8389282, 279.3439636, -186.7167969, 468.4582214, -585.2971191, 466.0607300
4: -312.4699707, 226.3032074, -532.9117432, 361.6045532, -674.0745239, 759.2149048

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.3323501, upper bound: 807.9630414
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5942547, upper bound: 807.9812257
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -313.7936096, 275.9430542, -307.9834900, 263.6258240, -577.4194336, 583.9265137
1: -250.9034882, 267.1716003, -245.6811523, 257.2617493, -508.1652222, 512.8527832
2: -365.3001404, 292.2919312, -355.0937805, 282.0675049, -647.3676758, 647.3857422
3: -147.6793365, 362.8142700, -143.2843018, 352.3505249, -500.0297546, 506.0985413
4: -407.4484558, 290.0770569, -396.7920837, 278.2794495, -685.7279053, 686.8691406

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7701661, upper bound: 806.2163520
time: 0.61 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5424493, upper bound: 806.2167479
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -431.8350830, 361.4309082, -328.4419861, 278.3179321, -710.1530151, 689.8729248
1: -346.1146851, 350.5698242, -262.1641235, 271.5196838, -617.6343384, 612.7338867
2: -502.6436768, 382.8444824, -378.9097290, 297.3813171, -800.0249023, 761.7542114
3: -195.2737427, 492.6449280, -151.7941589, 374.8180542, -570.0917969, 644.4390869
4: -560.0805664, 379.1239624, -423.3049011, 293.4226990, -853.5032959, 802.4288330

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7773715, upper bound: 805.9768533
time: 0.54 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8480100, upper bound: 803.8545165
time: 0.58 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -360.6799011, 304.4330139, -362.6666565, 305.9226074, -666.6024780, 667.0996094
1: -288.5881348, 295.3973083, -290.2025757, 296.8375549, -585.4256592, 585.5998535
2: -419.0789490, 322.8586731, -421.4447021, 324.4236450, -743.5025024, 744.3033447
3: -163.9579620, 411.7017212, -164.7423096, 413.9035950, -577.8615723, 576.4440308
4: -466.9795837, 319.7095337, -469.5854187, 321.2560120, -788.2355957, 789.2949219

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8330768, upper bound: 806.2771886
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8331937, upper bound: 806.4857756
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -360.6799011, 304.4330139, -428.7608032, 359.9247742, -720.6045532, 733.1936646
1: -288.5881348, 295.3973083, -343.3657837, 348.9992065, -637.5872803, 638.7630615
2: -419.0789490, 322.8586731, -499.1161499, 381.0668945, -800.1458130, 821.9748535
3: -163.9579620, 411.7017212, -194.1269379, 489.1353455, -653.0932617, 605.8286743
4: -466.9795837, 319.7095337, -556.5690918, 377.6802368, -844.6597900, 876.2786255

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7124686, upper bound: 806.6797948
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7124686, upper bound: 806.6797948
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -428.7608032, 359.9247742, -360.6799011, 304.4330139, -733.1936646, 720.6046143
1: -343.3657837, 348.9992065, -288.5881348, 295.3973083, -638.7630615, 637.5872803
2: -499.1161499, 381.0668945, -419.0789490, 322.8586731, -821.9748535, 800.1457520
3: -194.1269379, 489.1353455, -163.9579620, 411.7017212, -605.8286743, 653.0932617
4: -556.5690918, 377.6802368, -466.9795837, 319.7095337, -876.2786255, 844.6597900

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.6802081, upper bound: 806.7062846
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.6802081, upper bound: 806.7062836
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -428.7608032, 359.9247742, -427.0741272, 358.6000977, -787.3609009, 786.9987793
1: -343.3657837, 348.9992065, -341.9973145, 347.7138062, -691.0795288, 690.9965210
2: -499.1161499, 381.0668945, -497.1004333, 379.6651001, -878.7810669, 878.1673584
3: -194.1269379, 489.1353455, -193.4415894, 487.2424927, -681.3694458, 682.5768433
4: -556.5690918, 377.6802368, -554.3546143, 376.3097534, -932.8788452, 932.0348511

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.6798495, upper bound: 806.6797804
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.6798495, upper bound: 806.6797822
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -340.8095398, 290.1693420, -265.4413147, 234.5291290, -575.3385620, 555.6106567
1: -272.5678711, 281.5313416, -212.0245361, 227.3792267, -499.9469910, 493.5558777
2: -395.8618469, 307.7636108, -308.5409546, 248.7822266, -644.6439209, 616.3045654
3: -155.9651794, 389.7422485, -124.6852646, 307.7318726, -463.6970215, 514.4274292
4: -441.2347717, 304.8490295, -343.8453064, 246.8465729, -688.0812988, 648.6943359

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4580150, upper bound: 808.3287844
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4580150, upper bound: 808.3289309
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -361.0339355, 304.6775208, -368.1410522, 308.8850098, -669.9189453, 672.8186035
1: -288.8934326, 295.6718750, -294.9066162, 299.8633118, -588.7567139, 590.5784912
2: -419.5229187, 323.1707153, -427.6830444, 327.7184753, -747.2413330, 750.8536377
3: -164.0787048, 412.1364136, -166.5783997, 420.1256409, -584.2042847, 578.7148438
4: -467.4484863, 319.9922485, -475.9859619, 324.2613831, -791.7097778, 795.9781494

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4580150, upper bound: 808.3287844
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4580150, upper bound: 808.3289309
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -359.8309021, 303.4183350, -436.6898499, 365.1800842, -725.0109863, 740.1081543
1: -287.8851318, 294.4540100, -349.9454956, 354.0716248, -641.9567871, 644.3995361
2: -417.9902954, 321.8306274, -508.4296265, 386.5229187, -804.5131226, 830.2602539
3: -163.4290924, 410.5316467, -197.2324371, 497.9155884, -661.3446045, 607.7640381
4: -465.7576294, 318.6951904, -566.5614014, 382.8739319, -848.6314697, 885.2565918

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8433003, upper bound: 808.2841773
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8432116, upper bound: 808.3259405
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -424.6174011, 356.4237976, -436.6898499, 365.1800842, -789.7974854, 793.1136475
1: -339.9831543, 345.6016846, -349.9454956, 354.0716248, -694.0548096, 695.5470581
2: -494.1489258, 377.3796082, -508.4296265, 386.5229187, -880.6718140, 885.8092041
3: -192.2687531, 484.3070068, -197.2324371, 497.9155884, -690.1843262, 681.5394287
4: -551.0611572, 374.0391541, -566.5614014, 382.8739319, -933.9349976, 940.6005859

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8433003, upper bound: 808.2851018
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8432116, upper bound: 808.3268687
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -380.1496887, 318.2424622, -360.6799011, 304.4330139, -684.5825806, 678.9223633
1: -304.5241394, 308.7597961, -288.5881348, 295.3973083, -599.9214478, 597.3479004
2: -441.9195862, 337.3076477, -419.0789490, 322.8586731, -764.7782593, 756.3864136
3: -171.6442566, 433.2800598, -163.9579620, 411.7017212, -583.3459473, 597.2380371
4: -491.8168945, 333.9521179, -466.9795837, 319.7095337, -811.5264282, 800.9317017

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2342489, upper bound: 808.8322844
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7297508, upper bound: 806.7026970
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7297508, upper bound: 806.7030316
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -378.2431335, 316.8046570, -428.7608032, 359.9247742, -738.1679077, 745.5654297
1: -302.9781189, 307.3648682, -343.3657837, 348.9992065, -651.9772949, 650.7306519
2: -439.6520081, 335.7908936, -499.1161499, 381.0668945, -820.7188721, 834.9068604
3: -170.8855896, 431.1712341, -194.1269379, 489.1353455, -660.0208740, 625.2981567
4: -489.3193665, 332.4527588, -556.5690918, 377.6802368, -866.9996338, 889.0218506

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7073410, upper bound: 806.6746237
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7073410, upper bound: 806.6746237
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -436.6898499, 365.1800842, -359.8309021, 303.4183350, -740.1081543, 725.0109863
1: -349.9454956, 354.0716248, -287.8851318, 294.4540100, -644.3995361, 641.9567871
2: -508.4296265, 386.5229187, -417.9902954, 321.8306274, -830.2602539, 804.5131226
3: -197.2324371, 497.9155884, -163.4290924, 410.5316467, -607.7640381, 661.3446045
4: -566.5614014, 382.8739319, -465.7576294, 318.6951904, -885.2565308, 848.6314087

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.0873098, upper bound: 808.3424130
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3252916, upper bound: 808.3566167
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -436.6898499, 365.1800842, -424.6174011, 356.4237976, -793.1136475, 789.7974854
1: -349.9454956, 354.0716248, -339.9831543, 345.6016846, -695.5470581, 694.0548096
2: -508.4296265, 386.5229187, -494.1489258, 377.3796082, -885.8091431, 880.6718140
3: -197.2324371, 497.9155884, -192.2687531, 484.3070068, -681.5394287, 690.1843262
4: -566.5614014, 382.8739319, -551.0611572, 374.0391541, -940.6005859, 933.9349976

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.0869272, upper bound: 808.3424062
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3252913, upper bound: 808.3566099
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -356.1221313, 298.3259277, -401.6818542, 336.6027527, -692.7248535, 700.0078125
1: -284.9533386, 289.6357117, -321.4362793, 326.5682068, -611.5215454, 611.0719604
2: -413.1746521, 316.7574463, -466.4032898, 357.0509338, -770.2254639, 783.1607666
3: -161.4242706, 405.3092346, -182.3295593, 457.2983093, -618.7225952, 587.6387329
4: -460.0152283, 313.3795776, -519.8906250, 353.3977356, -813.4129639, 833.2701416

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2015798, upper bound: 808.2024857
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2020788, upper bound: 808.2024857
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -411.4728699, 344.4328003, -419.5442810, 351.3311157, -762.8039551, 763.9770508
1: -329.3374329, 334.1509399, -335.8997192, 340.8306885, -670.1680908, 670.0506592
2: -478.0291443, 365.2938538, -487.5963440, 372.4834900, -850.5126343, 852.8901978
3: -186.7167969, 468.4582214, -190.3561554, 477.7799683, -664.4967651, 658.8143921
4: -532.9117432, 361.6045532, -543.4680786, 368.7157288, -901.6273804, 905.0726318

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2020788, upper bound: 808.2024857
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2015798, upper bound: 808.2024857
time: 0.64 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 1.93 seconds
NS_A1_B2_A1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -805.5665718, upper bound: 807.6211005
NS_A1_B2_A1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -805.5665719, upper bound: 807.9909316
NS_A1_B2_A1_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -805.3323501, upper bound: 807.9170288
NS_A1_B2_A1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -805.5942547, upper bound: 807.8635990
NS_A1_B2_A1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -805.3323501, upper bound: 807.9630414
NS_A1_B2_A1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -805.5942547, upper bound: 807.9812257
NS_A2_B1_B2_B1_A2_B2_A1_A1, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 4, lower bound: -806.7701661, upper bound: 806.2163520
NS_A2_B1_B2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -807.5424493, upper bound: 806.2167479
NS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -808.8330768, upper bound: 806.2771886
NS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -808.8331937, upper bound: 806.4857756
NS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 4, lower bound: -806.7124686, upper bound: 806.6797948
NS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 4, lower bound: -806.7124686, upper bound: 806.6797948
NS_A2_B2_A1_B1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 4, lower bound: -806.6802081, upper bound: 806.7062846
NS_A2_B2_A1_B1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 4, lower bound: -806.6802081, upper bound: 806.7062836
NS_A2_B2_A1_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 4, lower bound: -806.6798495, upper bound: 806.6797804
NS_A2_B2_A1_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 4, lower bound: -806.6798495, upper bound: 806.6797822
NS_A2_B2_A1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -808.4580150, upper bound: 808.3287844
NS_A2_B2_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -808.4580150, upper bound: 808.3289309
NS_A2_B2_A1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -808.4580150, upper bound: 808.3287844
NS_A2_B2_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -808.4580150, upper bound: 808.3289309
NS_A2_B2_A1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -808.8433003, upper bound: 808.2841773
NS_A2_B2_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -808.8432116, upper bound: 808.3259405
NS_A2_B2_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -808.8433003, upper bound: 808.2851018
NS_A2_B2_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -808.8432116, upper bound: 808.3268687
NS_A2_B2_A2_B1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 4, lower bound: -806.7297508, upper bound: 806.7026970
NS_A2_B2_A2_B1_A1_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 4, lower bound: -806.7297508, upper bound: 806.7030316
NS_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 4, lower bound: -806.7073410, upper bound: 806.6746237
NS_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.93
Output dim: 4, lower bound: -806.7073410, upper bound: 806.6746237
NS_A2_B2_A2_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -807.0873098, upper bound: 808.3424130
NS_A2_B2_A2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -808.3252916, upper bound: 808.3566167
NS_A2_B2_A2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -807.0869272, upper bound: 808.3424062
NS_A2_B2_A2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -808.3252913, upper bound: 808.3566099
NS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -808.2015798, upper bound: 808.2024857
NS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -808.2020788, upper bound: 808.2024857
NS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -808.2020788, upper bound: 808.2024857
NS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.93
Output dim: 4, lower bound: -808.2015798, upper bound: 808.2024857

## BFS NS instance: NS_A1_B2_A1_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -262.4254456, 223.5715485, -359.9024353, 301.8214111, -564.2468262, 583.4739990
1: -208.6590271, 218.4313049, -288.0487671, 292.9872437, -501.6462708, 506.4800720
2: -300.1813049, 239.5306091, -417.8614807, 320.4252625, -620.6065674, 657.3920288
3: -123.2949066, 297.3791199, -163.2777863, 409.8174438, -533.1123657, 460.6568909
4: -335.5886230, 236.8408356, -465.2668457, 316.9648132, -652.5534058, 702.1076660

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.3994219, upper bound: 806.1424544
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.3370502, upper bound: 806.1423690
time: 1.36 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -314.6780396, 267.6726379, -359.9024353, 301.8214111, -616.4994507, 627.5750732
1: -250.8013306, 260.3953247, -288.0487671, 292.9872437, -543.7885742, 548.4440918
2: -361.2578735, 285.3591919, -417.8614807, 320.4252625, -681.6831055, 703.2206421
3: -147.9338989, 357.9339294, -163.2777863, 409.8174438, -557.7512207, 521.2116699
4: -403.9733582, 282.7382507, -465.2668457, 316.9648132, -720.9381714, 748.0051270

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A2_A2_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.3994219, upper bound: 806.1424544
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A2_A2_A2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.3370502, upper bound: 806.1423690
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -216.2514648, 192.2908783, -395.3339844, 332.9723511, -549.2236938, 587.6248779
1: -171.6279144, 188.0630341, -316.1922607, 323.0583801, -494.6862793, 504.2552490
2: -246.7999725, 206.4940338, -459.2079468, 353.2919312, -600.0919189, 665.7019043
3: -105.3072586, 248.2400818, -180.3746643, 450.6060486, -555.9132080, 428.6147461
4: -276.9530334, 204.1986389, -512.4122314, 349.9013367, -626.8543701, 716.6108398

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.3321533, upper bound: 807.2592604
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.3321533, upper bound: 807.8635842
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -325.3053284, 275.1112061, -400.0057678, 336.5234070, -661.8285522, 675.1169434
1: -259.4033203, 267.5791931, -319.9235840, 326.5131226, -585.9164429, 587.5028076
2: -373.8045959, 293.0217590, -464.6125793, 357.1040344, -730.9086304, 757.6343384
3: -151.9472504, 369.8461304, -182.3023987, 455.7301636, -607.6773071, 552.1485596
4: -417.9171143, 290.3411560, -518.3915405, 353.6432190, -771.5602417, 808.7326660

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.5940333, upper bound: 807.2592604
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5940333, upper bound: 807.8635842
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -216.2514648, 192.2908783, -404.3865662, 339.0111389, -555.2625122, 596.6774292
1: -171.6279144, 188.0630341, -323.6445618, 328.8689575, -500.4968872, 511.7075195
2: -246.7999725, 206.4940338, -469.7869873, 359.4902039, -606.2901611, 676.2810059
3: -105.3072586, 248.2400818, -183.7893524, 460.6073608, -565.9145508, 432.0294189
4: -276.9530334, 204.1986389, -523.8207397, 355.9425049, -632.8955078, 728.0193481

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.3318713, upper bound: 806.8392531
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.3318713, upper bound: 807.9630414
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -325.3053284, 275.1112061, -408.9078064, 342.4141235, -667.7192383, 684.0189209
1: -259.4033203, 267.5791931, -327.2504578, 332.1754150, -591.5786133, 594.8296509
2: -373.8045959, 293.0217590, -474.9876709, 363.1389160, -736.9434814, 768.0093994
3: -151.9472504, 369.8461304, -185.6300659, 465.5311890, -617.4784546, 555.4761963
4: -417.9171143, 290.3411560, -529.5704956, 359.5021973, -777.4192505, 819.9116211

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5664776, upper bound: 807.4477188
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5664776, upper bound: 807.9812289
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -306.5357361, 269.7943420, -307.9834900, 263.6258240, -570.1614990, 577.7777710
1: -245.0072632, 261.2319641, -245.6811523, 257.2617493, -502.2690125, 506.9131165
2: -356.7266846, 285.8985901, -355.0937805, 282.0675049, -638.7941895, 640.9923706
3: -144.4567566, 354.4788208, -143.2843018, 352.3505249, -496.8072205, 497.7631226
4: -397.9586182, 283.7874451, -396.7920837, 278.2794495, -676.2380371, 680.5795288

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5252448, upper bound: 805.9762696
time: 0.52 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -338.6922302, 288.5703125, -238.8508759, 215.9740753, -554.6662598, 527.4212036
1: -270.8442688, 279.9792786, -190.3831024, 209.4111938, -480.2554626, 470.3623657
2: -393.3395691, 306.0799561, -277.0705872, 229.3539886, -622.6934814, 583.1505127
3: -155.1239471, 387.3755493, -114.6507111, 278.3254089, -433.4493103, 502.0262146
4: -438.4590454, 303.1834412, -309.5355530, 227.6612244, -666.1202393, 612.7189331

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4330068, upper bound: 806.2763095
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4330068, upper bound: 806.2771886
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -359.0892944, 303.2238464, -350.1861877, 296.3976135, -655.4868164, 653.4100342
1: -287.3128967, 294.2649231, -280.1870117, 287.8131714, -575.1260986, 574.4519043
2: -417.2072144, 321.6420288, -406.6942444, 314.7042542, -731.9114990, 728.3363037
3: -163.3130035, 409.9819946, -159.5774536, 400.3084106, -563.6213379, 569.5593872
4: -464.9008179, 318.4823914, -453.2059937, 311.4501648, -776.3508911, 771.6883545

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4330996, upper bound: 806.4847707
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4330996, upper bound: 806.4857760
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -238.8508759, 215.9740753, -265.4413147, 234.5291290, -473.3799133, 481.4154053
1: -190.3831024, 209.4111938, -212.0245361, 227.3792267, -417.7622681, 421.4357300
2: -277.0705872, 229.3539886, -308.5409546, 248.7822266, -525.8527832, 537.8947754
3: -114.6507111, 278.3254089, -124.6852646, 307.7318726, -422.3825378, 403.0106506
4: -309.5355530, 227.6612244, -343.8453064, 246.8465729, -556.3821411, 571.5065308

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## BFS NS instance: NS_A2_B2_A1_B2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -350.1861877, 296.3976135, -265.4413147, 234.5291290, -584.7153320, 561.8388672
1: -280.1870117, 287.8131714, -212.0245361, 227.3792267, -507.5661316, 499.8377075
2: -406.6942444, 314.7042542, -308.5409546, 248.7822266, -655.4764404, 623.2450562
3: -159.5774536, 400.3084106, -124.6852646, 307.7318726, -467.3093262, 524.9936523
4: -453.2059937, 311.4501648, -343.8453064, 246.8465729, -700.0524292, 655.2954102

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A1_B2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -238.8508759, 215.9740753, -368.1410522, 308.8850098, -547.7357788, 584.1151123
1: -190.3831024, 209.4111938, -294.9066162, 299.8633118, -490.2463989, 504.3178101
2: -277.0705872, 229.3539886, -427.6830444, 327.7184753, -604.7890625, 657.0369263
3: -114.6507111, 278.3254089, -166.5783997, 420.1256409, -534.7763062, 444.9037476
4: -309.5355530, 227.6612244, -475.9859619, 324.2613831, -633.7968140, 703.6472168

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A1_B2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -350.1861877, 296.3976135, -368.1410522, 308.8850098, -659.0711670, 664.5386963
1: -280.1870117, 287.8131714, -294.9066162, 299.8633118, -580.0502930, 582.7196655
2: -406.6942444, 314.7042542, -427.6830444, 327.7184753, -734.4127197, 742.3872070
3: -159.5774536, 400.3084106, -166.5783997, 420.1256409, -579.7030640, 566.8867798
4: -453.2059937, 311.4501648, -475.9859619, 324.2613831, -777.4671631, 787.4359741

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -338.1403809, 287.7702026, -301.9802551, 266.9254761, -605.0657349, 589.7504272
1: -270.3888245, 279.2289429, -241.3724365, 258.4585876, -528.8473511, 520.6013794
2: -392.6057129, 305.2544861, -351.4915771, 282.9038696, -675.5095215, 656.7458496
3: -154.7328949, 386.5610962, -142.9042206, 349.7152405, -504.4481201, 529.4651489
4: -437.6303101, 302.3576355, -392.2167969, 280.8281860, -718.4584961, 694.5742188

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4845259, upper bound: 808.2561996
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4581953, upper bound: 808.2840550
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4581953, upper bound: 808.2841773
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -358.1225586, 302.1332397, -425.2669678, 355.6134644, -713.7360229, 727.4000854
1: -286.5150146, 293.2277832, -340.7726135, 345.0171204, -631.5321045, 634.0003662
2: -415.9786377, 320.5117798, -494.8684692, 376.8844604, -792.8630981, 815.3801880
3: -162.7322540, 408.6770325, -192.2481842, 485.1243896, -647.8565674, 600.9252319
4: -463.5232544, 317.3655701, -551.5020752, 373.2102356, -836.7335205, 868.8674927

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4586631, upper bound: 808.3258182
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4586631, upper bound: 808.3259405
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -401.5859375, 339.6194153, -301.9802551, 266.9254761, -668.5111694, 641.5996704
1: -321.3681030, 329.2499695, -241.3724365, 258.4585876, -579.8265381, 570.6224365
2: -467.2197876, 359.7701721, -351.4915771, 282.9038696, -750.1235352, 711.2616577
3: -182.9760895, 458.8687744, -142.9042206, 349.7152405, -532.6913452, 601.7728882
4: -521.0924683, 356.6342773, -392.2167969, 280.8281860, -801.9205933, 748.8508301

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4840760, upper bound: 808.2561511
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -423.1084290, 355.1769104, -425.2669678, 355.6134644, -778.7219238, 780.4438477
1: -338.7695007, 344.4127808, -340.7726135, 345.0171204, -683.7866211, 685.1853027
2: -492.3632202, 376.0985718, -494.8684692, 376.8844604, -869.2476807, 870.9670410
3: -191.6167755, 482.6070557, -192.2481842, 485.1243896, -676.7411499, 674.8552246
4: -549.0739746, 372.7546387, -551.5020752, 373.2102356, -922.2841187, 924.2565918

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4840073, upper bound: 808.2985472
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2656078, upper bound: 808.3265394
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9425914, upper bound: 808.3266639
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -301.9802551, 266.9254761, -338.1403809, 287.7702026, -589.7504883, 605.0657349
1: -241.3724365, 258.4585876, -270.3888245, 279.2289429, -520.6013794, 528.8473511
2: -351.4915771, 282.9038696, -392.6057129, 305.2544861, -656.7458496, 675.5095215
3: -142.9042206, 349.7152405, -154.7328949, 386.5610962, -529.4651489, 504.4481201
4: -392.2167969, 280.8281860, -437.6303101, 302.3576355, -694.5742188, 718.4584961

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -425.2669678, 355.6134644, -358.1225586, 302.1332397, -727.4001465, 713.7360229
1: -340.7726135, 345.0171204, -286.5150146, 293.2277832, -634.0003662, 631.5321045
2: -494.8684692, 376.8844604, -415.9786377, 320.5117798, -815.3801880, 792.8630981
3: -192.2481842, 485.1243896, -162.7322540, 408.6770325, -600.9252319, 647.8565674
4: -551.5020752, 373.2102356, -463.5232544, 317.3655701, -868.8676147, 836.7335205

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -301.9802551, 266.9254761, -401.5859375, 339.6194153, -641.5996704, 668.5111694
1: -241.3724365, 258.4585876, -321.3681030, 329.2499695, -570.6224365, 579.8265991
2: -351.4915771, 282.9038696, -467.2197876, 359.7701721, -711.2616577, 750.1235962
3: -142.9042206, 349.7152405, -182.9760895, 458.8687744, -601.7729492, 532.6913452
4: -392.2167969, 280.8281860, -521.0924683, 356.6342773, -748.8508911, 801.9205933

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -425.2669678, 355.6134644, -423.1084290, 355.1769104, -780.4438477, 778.7219238
1: -340.7726135, 345.0171204, -338.7695007, 344.4127808, -685.1853638, 683.7866211
2: -494.8684692, 376.8844604, -492.3632202, 376.0985718, -870.9670410, 869.2476807
3: -192.2481842, 485.1243896, -191.6167755, 482.6070557, -674.8552246, 676.7411499
4: -551.5020752, 373.2102356, -549.0739746, 372.7546387, -924.2565308, 922.2841797

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.0113354, upper bound: 807.0361512
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3251680, upper bound: 808.3561161
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -356.1221313, 298.3259277, -356.1221313, 298.3259277, -654.4480591, 654.4480591
1: -284.9533386, 289.6357117, -284.9533386, 289.6357117, -574.5890503, 574.5890503
2: -413.1746521, 316.7574463, -413.1746521, 316.7574463, -729.9321289, 729.9321289
3: -161.4242706, 405.3092346, -161.4242706, 405.3092346, -566.7335205, 566.7335205
4: -460.0152283, 313.3795776, -460.0152283, 313.3795776, -773.3947144, 773.3947144

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1514889, upper bound: 808.0969944
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1757294, upper bound: 808.1585532
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -356.1221313, 298.3259277, -411.4728699, 344.4328003, -700.5549316, 709.7988281
1: -284.9533386, 289.6357117, -329.3374329, 334.1509399, -619.1042480, 618.9731445
2: -413.1746521, 316.7574463, -478.0291443, 365.2938538, -778.4685059, 794.7866211
3: -161.4242706, 405.3092346, -186.7167969, 468.4582214, -629.8825073, 592.0260010
4: -460.0152283, 313.3795776, -532.9117432, 361.6045532, -821.6197510, 846.2912598

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1514889, upper bound: 808.0969944
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1757294, upper bound: 808.1585532
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -411.4728699, 344.4328003, -356.1221313, 298.3259277, -709.7988281, 700.5549316
1: -329.3374329, 334.1509399, -284.9533386, 289.6357117, -618.9731445, 619.1042480
2: -478.0291443, 365.2938538, -413.1746521, 316.7574463, -794.7866211, 778.4685059
3: -186.7167969, 468.4582214, -161.4242706, 405.3092346, -592.0260010, 629.8825073
4: -532.9117432, 361.6045532, -460.0152283, 313.3795776, -846.2913208, 821.6197510

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0964942, upper bound: 808.1199575
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1576061, upper bound: 808.1582333
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -411.4728699, 344.4328003, -411.4728699, 344.4328003, -755.9056396, 755.9056396
1: -329.3374329, 334.1509399, -329.3374329, 334.1509399, -663.4884033, 663.4884033
2: -478.0291443, 365.2938538, -478.0291443, 365.2938538, -843.3229980, 843.3229980
3: -186.7167969, 468.4582214, -186.7167969, 468.4582214, -655.1750488, 655.1750488
4: -532.9117432, 361.6045532, -532.9117432, 361.6045532, -894.5162964, 894.5162964

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0964942, upper bound: 808.1199573
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1576061, upper bound: 808.1582333
time: 0.61 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 2.37 seconds
NS_A1_B2_A1_A2_B2_B1_A2_A1_A1, status: Status.VERIFIED, split count: 9, time: 2.37
Output dim: 4, lower bound: -805.3994219, upper bound: 806.1424544
NS_A1_B2_A1_A2_B2_B1_A2_A1_A2, status: Status.VERIFIED, split count: 9, time: 2.37
Output dim: 4, lower bound: -805.3370502, upper bound: 806.1423690
NS_A1_B2_A1_A2_B2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 9, time: 2.37
Output dim: 4, lower bound: -805.3994219, upper bound: 806.1424544
NS_A1_B2_A1_A2_B2_B1_A2_A2_A2, status: Status.VERIFIED, split count: 9, time: 2.37
Output dim: 4, lower bound: -805.3370502, upper bound: 806.1423690
NS_A1_B2_A1_A2_B2_B2_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 2.37
Output dim: 4, lower bound: -805.3321533, upper bound: 807.2592604
NS_A1_B2_A1_A2_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -805.3321533, upper bound: 807.8635842
NS_A1_B2_A1_A2_B2_B2_B1_A2_B1, status: Status.VERIFIED, split count: 9, time: 2.37
Output dim: 4, lower bound: -805.5940333, upper bound: 807.2592604
NS_A1_B2_A1_A2_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -805.5940333, upper bound: 807.8635842
NS_A1_B2_A1_A2_B2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 2.37
Output dim: 4, lower bound: -805.3318713, upper bound: 806.8392531
NS_A1_B2_A1_A2_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -805.3318713, upper bound: 807.9630414
NS_A1_B2_A1_A2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -805.5664776, upper bound: 807.4477188
NS_A1_B2_A1_A2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -805.5664776, upper bound: 807.9812289
NS_A2_B2_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.4330068, upper bound: 806.2763095
NS_A2_B2_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.4330068, upper bound: 806.2771886
NS_A2_B2_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.4330996, upper bound: 806.4847707
NS_A2_B2_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.4330996, upper bound: 806.4857760
NS_A2_B2_A1_B2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.4581953, upper bound: 808.2840550
NS_A2_B2_A1_B2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.4581953, upper bound: 808.2841773
NS_A2_B2_A1_B2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.4586631, upper bound: 808.3258182
NS_A2_B2_A1_B2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.4586631, upper bound: 808.3259405
NS_A2_B2_A1_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.2656078, upper bound: 808.3265394
NS_A2_B2_A1_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.9425914, upper bound: 808.3266639
NS_A2_B2_A2_B1_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 9, time: 2.37
Output dim: 4, lower bound: -807.0113354, upper bound: 807.0361512
NS_A2_B2_A2_B1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.3251680, upper bound: 808.3561161
NS_A2_B2_A2_B2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.1514889, upper bound: 808.0969944
NS_A2_B2_A2_B2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.1757294, upper bound: 808.1585532
NS_A2_B2_A2_B2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.1514889, upper bound: 808.0969944
NS_A2_B2_A2_B2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.1757294, upper bound: 808.1585532
NS_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.0964942, upper bound: 808.1199575
NS_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.1576061, upper bound: 808.1582333
NS_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.0964942, upper bound: 808.1199573
NS_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.37
Output dim: 4, lower bound: -808.1576061, upper bound: 808.1582333

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -216.2514648, 192.2908783, -473.8027954, 394.5919189, -610.8432617, 666.0936890
1: -171.6279144, 188.0630341, -379.6530762, 382.5795593, -554.2073364, 567.7161255
2: -246.7999725, 206.4940338, -551.1073608, 417.8556213, -664.6555176, 757.6013794
3: -105.3072586, 248.2400818, -215.2121735, 538.6518555, -643.9590454, 463.4522705
4: -276.9530334, 204.1986389, -614.4487305, 414.4178772, -691.3709106, 818.6473389

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -804.8807620, upper bound: 807.7207461
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -325.3053284, 275.1112061, -473.8817749, 394.6433411, -719.9484863, 748.9929810
1: -259.4033203, 267.5791931, -379.7167358, 382.6298218, -642.0329590, 647.2958374
2: -373.8045959, 293.0217590, -551.2014771, 417.9100952, -791.7147217, 844.2232666
3: -151.9472504, 369.8461304, -215.2387848, 538.7371826, -690.6843872, 585.0848999
4: -417.9171143, 290.3411560, -614.5518799, 414.4721985, -832.3892212, 904.8930664

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.5065838, upper bound: 807.2238978
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -216.2514648, 192.2908783, -482.8277588, 400.5331726, -616.7844849, 675.1186523
1: -171.6279144, 188.0630341, -387.0942078, 388.4757080, -560.1035767, 575.1571045
2: -246.7999725, 206.4940338, -561.7634277, 424.2043457, -671.0043335, 768.2573853
3: -105.3072586, 248.2400818, -218.3296814, 548.8738403, -654.1810913, 466.5697632
4: -276.9530334, 204.1986389, -625.9031372, 420.5724487, -697.5255127, 830.1017456

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.4474598, upper bound: 806.9321084
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -804.9126683, upper bound: 807.7502614
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -262.4254456, 223.5715485, -408.9078064, 342.4141235, -604.8395386, 632.4791870
1: -208.6590271, 218.4313049, -327.2504578, 332.1754150, -540.8342896, 545.6817627
2: -300.1813049, 239.5306091, -474.9876709, 363.1389160, -663.3201904, 714.5183105
3: -123.2949066, 297.3791199, -185.6300659, 465.5311890, -588.8261108, 483.0091858
4: -335.5886230, 236.8408356, -529.5704956, 359.5021973, -695.0908203, 766.4113159

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.3379726, upper bound: 807.2785029
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.2630670, upper bound: 807.3923736
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -321.6694641, 272.1740417, -408.9078064, 342.4141235, -664.0836182, 681.0817261
1: -256.4455872, 264.6746521, -327.2504578, 332.1754150, -588.6208496, 591.9249878
2: -369.4765930, 289.9022217, -474.9876709, 363.1389160, -732.6154175, 764.8898926
3: -150.4444580, 365.6299438, -185.6300659, 465.5311890, -615.9756470, 551.2600098
4: -413.1534729, 287.3654785, -529.5704956, 359.5021973, -772.6556396, 816.9359741

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.2630670, upper bound: 807.7569442
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -236.7261505, 214.4211121, -238.8508759, 215.9740753, -452.7002258, 453.2719727
1: -188.6264191, 207.9092102, -190.3831024, 209.4111938, -398.0375977, 398.2922363
2: -274.4982300, 227.7375488, -277.0705872, 229.3539886, -503.8521423, 504.8081055
3: -113.8412552, 275.9186707, -114.6507111, 278.3254089, -392.1666260, 390.5693970
4: -306.7277832, 226.0623169, -309.5355530, 227.6612244, -534.3889771, 535.5977783

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4330155, upper bound: 806.2763095
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4330155, upper bound: 806.2763095
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -348.3787842, 295.0643921, -238.8508759, 215.9740753, -564.3527832, 533.9152832
1: -278.7173157, 286.5200195, -190.3831024, 209.4111938, -488.1285095, 476.9030457
2: -404.5417480, 313.2996521, -277.0705872, 229.3539886, -633.8956299, 590.3702393
3: -158.8732300, 398.3055115, -114.6507111, 278.3254089, -437.1986389, 512.9561157
4: -450.8406067, 310.0665588, -309.5355530, 227.6612244, -678.5018311, 619.6021118

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4330155, upper bound: 806.2771886
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4330154, upper bound: 806.2771886
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -236.7261505, 214.4211121, -350.1861877, 296.3976135, -533.1237183, 564.6072998
1: -188.6264191, 207.9092102, -280.1870117, 287.8131714, -476.4395752, 488.0961304
2: -274.4982300, 227.7375488, -406.6942444, 314.7042542, -589.2023926, 634.4317627
3: -113.8412552, 275.9186707, -159.5774536, 400.3084106, -514.1496582, 435.4961243
4: -306.7277832, 226.0623169, -453.2059937, 311.4501648, -618.1776733, 679.2683105

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4330068, upper bound: 806.4847700
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4330068, upper bound: 806.4847700
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -348.3787842, 295.0643921, -350.1861877, 296.3976135, -644.7762451, 645.2506104
1: -278.7173157, 286.5200195, -280.1870117, 287.8131714, -566.5304565, 566.7070312
2: -404.5417480, 313.2996521, -406.6942444, 314.7042542, -719.2459106, 719.9938965
3: -158.8732300, 398.3055115, -159.5774536, 400.3084106, -559.1816406, 557.8828735
4: -450.8406067, 310.0665588, -453.2059937, 311.4501648, -762.2905884, 763.2725220

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4330068, upper bound: 806.4857756
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4330068, upper bound: 806.4857756
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -236.2806091, 213.6303864, -301.9802551, 266.9254761, -503.2060242, 515.6105957
1: -188.2987976, 207.1317749, -241.3724365, 258.4585876, -446.7573242, 448.5042114
2: -273.9974365, 226.8723755, -351.4915771, 282.9038696, -556.9011841, 578.3638916
3: -113.3962555, 275.3093262, -142.9042206, 349.7152405, -463.1115112, 418.2135315
4: -306.1318970, 225.2442169, -392.2167969, 280.8281860, -586.9600220, 617.4609985

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -347.2870789, 293.8791504, -301.9802551, 266.9254761, -614.2125244, 595.8593750
1: -277.8208923, 285.3752136, -241.3724365, 258.4585876, -536.2794800, 526.7476196
2: -403.1630859, 312.0485840, -351.4915771, 282.9038696, -686.0669556, 663.5401001
3: -158.2340240, 396.8710632, -142.9042206, 349.7152405, -507.9492798, 539.7752686
4: -449.2938843, 308.8259888, -392.2167969, 280.8281860, -730.1220703, 701.0427246

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -236.2806091, 213.6303864, -425.2669678, 355.6134644, -591.8940430, 638.8973389
1: -188.2987976, 207.1317749, -340.7726135, 345.0171204, -533.3158569, 547.9042969
2: -273.9974365, 226.8723755, -494.8684692, 376.8844604, -650.8818970, 721.7407837
3: -113.3962555, 275.3093262, -192.2481842, 485.1243896, -598.5206299, 467.5574951
4: -306.1318970, 225.2442169, -551.5020752, 373.2102356, -679.3421631, 776.7462769

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -347.2870789, 293.8791504, -425.2669678, 355.6134644, -702.9005127, 719.1461182
1: -277.8208923, 285.3752136, -340.7726135, 345.0171204, -622.8380127, 626.1477661
2: -403.1630859, 312.0485840, -494.8684692, 376.8844604, -780.0475464, 806.9170532
3: -158.2340240, 396.8710632, -192.2481842, 485.1243896, -643.3583984, 589.1192627
4: -449.2938843, 308.8259888, -551.5020752, 373.2102356, -822.5040894, 860.3280640

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -389.5967712, 330.1230164, -417.8737488, 350.1210938, -739.7178955, 747.9966431
1: -311.8061523, 319.9088135, -334.8337097, 339.5743408, -651.3804932, 654.7424316
2: -453.2441406, 349.2494812, -486.2662048, 370.8413086, -824.0853271, 835.5156860
3: -177.4509583, 445.4598083, -189.1735382, 476.9936523, -654.4445801, 634.6333008
4: -505.9409180, 346.5921936, -542.0020752, 367.3201599, -873.2608032, 888.5942383

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1384524, upper bound: 807.0104477
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1384524, upper bound: 808.3265394
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -494.7641296, 410.9326782, -422.7106323, 353.6160583, -848.3801880, 833.6433105
1: -396.7264709, 398.4463501, -338.6902466, 343.0493774, -739.7758179, 737.1365967
2: -576.2191772, 434.8451538, -491.8343506, 374.7344666, -950.9536133, 926.6795044
3: -223.4765930, 562.8938599, -191.1598206, 482.2039185, -705.6802979, 754.0537109
4: -642.2418823, 431.4424744, -548.1655273, 371.1097412, -1013.3516235, 979.6079712

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3203895, upper bound: 807.0105402
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3203895, upper bound: 808.3266639
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -422.7106323, 353.6160583, -494.7641296, 410.9326782, -833.6433105, 848.3801880
1: -338.6902466, 343.0493774, -396.7264709, 398.4463501, -737.1365967, 739.7758789
2: -491.8343506, 374.7344666, -576.2191772, 434.8451538, -926.6795044, 950.9536133
3: -191.1598206, 482.2039185, -223.4765930, 562.8938599, -754.0536499, 705.6802368
4: -548.1655273, 371.1097412, -642.2418823, 431.4424744, -979.6080322, 1013.3516235

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.0097677, upper bound: 807.0382921
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.0097677, upper bound: 807.0382921
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -334.4406128, 282.4396057, -243.1517029, 214.5575256, -548.9980469, 525.5913086
1: -267.4475098, 274.2309875, -193.9788208, 208.4597931, -475.9071960, 468.2097778
2: -387.9226685, 300.1345215, -282.0811462, 228.5283966, -616.4510498, 582.2156372
3: -152.6686401, 381.3928223, -114.9791260, 281.8475647, -434.5162048, 496.3719482
4: -432.0298767, 296.9044495, -314.5017090, 226.3350677, -658.3649292, 611.4061279

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1756993, upper bound: 808.1752019
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1756993, upper bound: 808.1752019
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -354.5436707, 297.1626282, -343.6112366, 288.9629822, -643.5066528, 640.7738647
1: -283.6803284, 288.5280151, -274.8843079, 280.7315369, -564.4118652, 563.4123535
2: -411.2619019, 315.5614014, -398.1069336, 307.1251831, -718.3870850, 713.6682739
3: -160.7912903, 403.5765686, -156.3122864, 391.6330872, -552.4243774, 559.8887939
4: -457.8945007, 312.1682739, -443.3058777, 303.6446228, -761.5390625, 755.4741211

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1757294, upper bound: 808.1752019
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1757294, upper bound: 808.1752019
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -334.4406128, 282.4396057, -277.1587219, 244.9257812, -579.3663940, 559.5983276
1: -267.4475098, 274.2309875, -221.0946198, 237.6749420, -505.1224365, 495.3255615
2: -387.9226685, 300.1345215, -321.5402527, 260.5017395, -648.4244385, 621.6745605
3: -152.6686401, 381.3928223, -132.1236572, 320.6901550, -473.3587646, 513.5164795
4: -432.0298767, 296.9044495, -359.0209351, 258.1718445, -690.2017212, 655.9254150

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1514892, upper bound: 808.0969944
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1514891, upper bound: 808.0969944
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -354.5436707, 297.1626282, -399.4542847, 334.7368469, -689.2804565, 696.6169434
1: -283.6803284, 288.5280151, -319.7079773, 324.8833008, -608.5635986, 608.2359619
2: -411.2619019, 315.5614014, -463.7905884, 355.3358459, -766.5977783, 779.3519897
3: -160.7912903, 403.5765686, -181.5567322, 455.0501709, -615.8413696, 585.1333008
4: -457.8945007, 312.1682739, -517.1308594, 351.6369629, -809.5314331, 829.2991333

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1762122, upper bound: 808.1585337
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1762122, upper bound: 808.1585532
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -277.1587219, 244.9257812, -334.4406128, 282.4396057, -559.5983276, 579.3663940
1: -221.0946198, 237.6749420, -267.4475098, 274.2309875, -495.3255920, 505.1224365
2: -321.5402527, 260.5017395, -387.9226685, 300.1345215, -621.6746216, 648.4244385
3: -132.1236572, 320.6901550, -152.6686401, 381.3928223, -513.5164795, 473.3587646
4: -359.0209351, 258.1718445, -432.0298767, 296.9044495, -655.9253540, 690.2017212

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0964942, upper bound: 808.1092263
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0956420, upper bound: 808.1199575
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -399.4542847, 334.7368469, -354.5436707, 297.1626282, -696.6169434, 689.2804565
1: -319.7079773, 324.8833008, -283.6803284, 288.5280151, -608.2359619, 608.5635986
2: -463.7905884, 355.3358459, -411.2619019, 315.5614014, -779.3519897, 766.5977783
3: -181.5567322, 455.0501709, -160.7912903, 403.5765686, -585.1333008, 615.8413696
4: -517.1308594, 351.6369629, -457.8945007, 312.1682739, -829.2991333, 809.5314331

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0964942, upper bound: 808.1118359
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0956420, upper bound: 808.1583994
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -277.1587219, 244.9257812, -387.7337036, 326.7726135, -603.9313354, 632.6594849
1: -221.0946198, 237.6749420, -310.1421204, 317.0197144, -538.1143188, 547.8170776
2: -321.5402527, 260.5017395, -450.1676025, 346.7895813, -668.3296509, 710.6693115
3: -132.1236572, 320.6901550, -177.0679321, 442.1675415, -574.2910156, 497.7580566
4: -359.0209351, 258.1718445, -502.0212097, 343.2762451, -702.2971802, 760.1929932

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0956420, upper bound: 808.0970343
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0964942, upper bound: 808.1199573
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -399.4542847, 334.7368469, -410.0646667, 343.2792358, -742.7335205, 744.8015137
1: -319.7079773, 324.8833008, -328.2089844, 333.0480652, -652.7559814, 653.0922241
2: -463.7905884, 355.3358459, -476.3601685, 364.1110535, -827.9016113, 831.6960449
3: -181.5567322, 455.0501709, -186.1056213, 466.8699036, -648.4266357, 641.1557617
4: -517.1308594, 351.6369629, -531.0596313, 360.4205933, -877.5514526, 882.6965942

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0964942, upper bound: 808.0970371
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0962473, upper bound: 808.1582333
time: 0.68 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 2.16 seconds
NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.4330155, upper bound: 806.2763095
NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.4330155, upper bound: 806.2763095
NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.4330155, upper bound: 806.2771886
NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.4330154, upper bound: 806.2771886
NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.4330068, upper bound: 806.4847700
NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.4330068, upper bound: 806.4847700
NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.4330068, upper bound: 806.4857756
NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.4330068, upper bound: 806.4857756
NS_A2_B2_A1_B2_B2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.1384524, upper bound: 807.0104477
NS_A2_B2_A1_B2_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.1384524, upper bound: 808.3265394
NS_A2_B2_A1_B2_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.3203895, upper bound: 807.0105402
NS_A2_B2_A1_B2_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.3203895, upper bound: 808.3266639
NS_A2_B2_A2_B1_A2_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 10, time: 2.16
Output dim: 4, lower bound: -807.0097677, upper bound: 807.0382921
NS_A2_B2_A2_B1_A2_B2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 10, time: 2.16
Output dim: 4, lower bound: -807.0097677, upper bound: 807.0382921
NS_A2_B2_A2_B2_A2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.1756993, upper bound: 808.1752019
NS_A2_B2_A2_B2_A2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.1756993, upper bound: 808.1752019
NS_A2_B2_A2_B2_A2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.1757294, upper bound: 808.1752019
NS_A2_B2_A2_B2_A2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.1757294, upper bound: 808.1752019
NS_A2_B2_A2_B2_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.1514892, upper bound: 808.0969944
NS_A2_B2_A2_B2_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.1514891, upper bound: 808.0969944
NS_A2_B2_A2_B2_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.1762122, upper bound: 808.1585337
NS_A2_B2_A2_B2_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.1762122, upper bound: 808.1585532
NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.0964942, upper bound: 808.1092263
NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.0956420, upper bound: 808.1199575
NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.0964942, upper bound: 808.1118359
NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.0956420, upper bound: 808.1583994
NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.0956420, upper bound: 808.0970343
NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.0964942, upper bound: 808.1199573
NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.0964942, upper bound: 808.0970371
NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -808.0962473, upper bound: 808.1582333

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -236.7261505, 214.4211121, -236.7261505, 214.4211121, -451.1472778, 451.1472778
1: -188.6264191, 207.9092102, -188.6264191, 207.9092102, -396.5355225, 396.5355225
2: -274.4982300, 227.7375488, -274.4982300, 227.7375488, -502.2357788, 502.2357788
3: -113.8412552, 275.9186707, -113.8412552, 275.9186707, -389.7599182, 389.7599182
4: -306.7277832, 226.0623169, -306.7277832, 226.0623169, -532.7899170, 532.7899780

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -236.7261505, 214.4211121, -231.5045166, 211.1185455, -447.8446960, 445.9256287
1: -188.6264191, 207.9092102, -184.4334106, 204.8070679, -393.4334717, 392.3425293
2: -274.4982300, 227.7375488, -268.4873657, 224.2598419, -498.7580566, 496.2249146
3: -113.8412552, 275.9186707, -111.8914490, 270.5464783, -384.3877258, 387.8101196
4: -306.7277832, 226.0623169, -300.0113525, 222.6735840, -529.4013062, 526.0736694

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -348.3787842, 295.0643921, -236.7261505, 214.4211121, -562.7998047, 531.7905273
1: -278.7173157, 286.5200195, -188.6264191, 207.9092102, -486.6264038, 475.1463623
2: -404.5417480, 313.2996521, -274.4982300, 227.7375488, -632.2791748, 587.7978516
3: -158.8732300, 398.3055115, -113.8412552, 275.9186707, -434.7919006, 512.1467896
4: -450.8406067, 310.0665588, -306.7277832, 226.0623169, -676.9029541, 616.7943115

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -348.3787842, 295.0643921, -231.5045166, 211.1185455, -559.4971924, 526.5689087
1: -278.7173157, 286.5200195, -184.4334106, 204.8070679, -483.5243835, 470.9533691
2: -404.5417480, 313.2996521, -268.4873657, 224.2598419, -628.8015747, 581.7869873
3: -158.8732300, 398.3055115, -111.8914490, 270.5464783, -429.4197083, 510.1969299
4: -450.8406067, 310.0665588, -300.0113525, 222.6735840, -673.5140381, 610.0778809

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -236.7261505, 214.4211121, -348.3787842, 295.0643921, -531.7905273, 562.7998047
1: -188.6264191, 207.9092102, -278.7173157, 286.5200195, -475.1463623, 486.6264343
2: -274.4982300, 227.7375488, -404.5417480, 313.2996521, -587.7978516, 632.2792358
3: -113.8412552, 275.9186707, -158.8732300, 398.3055115, -512.1467285, 434.7919006
4: -306.7277832, 226.0623169, -450.8406067, 310.0665588, -616.7943115, 676.9028931

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -236.7261505, 214.4211121, -339.3971558, 288.9228821, -525.6490479, 553.8182373
1: -188.6264191, 207.9092102, -271.4136658, 280.6654663, -469.2918396, 479.3227844
2: -274.4982300, 227.7375488, -393.9730530, 306.8523865, -581.3505249, 621.7105103
3: -113.8412552, 275.9186707, -155.4444275, 388.4529724, -502.2942200, 431.3630676
4: -306.7277832, 226.0623169, -439.1615601, 303.7279663, -610.4556885, 665.2238770

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -348.3787842, 295.0643921, -348.3787842, 295.0643921, -643.4431763, 643.4431763
1: -278.7173157, 286.5200195, -278.7173157, 286.5200195, -565.2373047, 565.2373047
2: -404.5417480, 313.2996521, -404.5417480, 313.2996521, -717.8414307, 717.8414307
3: -158.8732300, 398.3055115, -158.8732300, 398.3055115, -557.1787109, 557.1787109
4: -450.8406067, 310.0665588, -450.8406067, 310.0665588, -760.9071655, 760.9071655

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -348.3787842, 295.0643921, -339.3971558, 288.9228821, -637.3016357, 634.4615479
1: -278.7173157, 286.5200195, -271.4136658, 280.6654663, -559.3828125, 557.9335938
2: -404.5417480, 313.2996521, -393.9730530, 306.8523865, -711.3940430, 707.2727051
3: -158.8732300, 398.3055115, -155.4444275, 388.4529724, -547.3261719, 553.7499390
4: -450.8406067, 310.0665588, -439.1615601, 303.7279663, -754.5686035, 749.2281494

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -389.5967712, 330.1230164, -391.0483398, 330.4823914, -720.0790405, 721.1712036
1: -311.8061523, 319.9088135, -313.2450562, 320.3414612, -632.1475830, 633.1538086
2: -453.2441406, 349.2494812, -454.9747925, 349.6975403, -802.9416504, 804.2241821
3: -177.4509583, 445.4598083, -178.0466003, 447.3768311, -624.8276978, 623.5062866
4: -505.9409180, 346.5921936, -507.4799805, 346.8109131, -852.7517090, 854.0721436

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -389.5967712, 330.1230164, -496.9978638, 412.1469727, -801.7436523, 827.1207275
1: -311.8061523, 319.9088135, -398.8172607, 399.7313843, -711.5375366, 718.7260742
2: -453.2441406, 349.2494812, -579.0264893, 436.1900024, -889.4341431, 928.2760010
3: -177.4509583, 445.4598083, -224.3730621, 566.0726318, -743.5234985, 669.8328857
4: -505.9409180, 346.5921936, -644.9775391, 432.5502930, -938.4909668, 991.5697021

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -494.7641296, 410.9326782, -391.0483398, 330.4823914, -825.2465210, 801.9809570
1: -396.7264709, 398.4463501, -313.2450562, 320.3414612, -717.0677490, 711.6914062
2: -576.2191772, 434.8451538, -454.9747925, 349.6975403, -925.9167480, 889.8198853
3: -223.4765930, 562.8938599, -178.0466003, 447.3768311, -670.8532715, 740.9404297
4: -642.2418823, 431.4424744, -507.4799805, 346.8109131, -989.0527954, 938.9223633

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -494.7641296, 410.9326782, -497.0875854, 412.2059021, -906.9699707, 908.0202637
1: -396.7264709, 398.4463501, -398.8895874, 399.7881470, -796.5144653, 797.3358154
2: -576.2191772, 434.8451538, -579.1336670, 436.2517700, -1012.4709473, 1013.9788208
3: -223.4765930, 562.8938599, -224.4025879, 566.1690674, -789.6455688, 787.2964478
4: -642.2418823, 431.4424744, -645.0947876, 432.6123657, -1074.8541260, 1076.5372314

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -243.1517029, 214.5575256, -243.1517029, 214.5575256, -457.7092285, 457.7092285
1: -193.9788208, 208.4597931, -193.9788208, 208.4597931, -402.4385376, 402.4385376
2: -282.0811462, 228.5283966, -282.0811462, 228.5283966, -510.6094971, 510.6095276
3: -114.9791260, 281.8475647, -114.9791260, 281.8475647, -396.8266602, 396.8266907
4: -314.5017090, 226.3350677, -314.5017090, 226.3350677, -540.8366699, 540.8366699

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -343.6112366, 288.9629822, -243.1517029, 214.5575256, -558.1687012, 532.1146851
1: -274.8843079, 280.7315369, -193.9788208, 208.4597931, -483.3440857, 474.7103577
2: -398.1069336, 307.1251831, -282.0811462, 228.5283966, -626.6353149, 589.2062988
3: -156.3122864, 391.6330872, -114.9791260, 281.8475647, -438.1598511, 506.6121521
4: -443.3058777, 303.6446228, -314.5017090, 226.3350677, -669.6408691, 618.1463013

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -243.1517029, 214.5575256, -343.6112366, 288.9629822, -532.1146851, 558.1687012
1: -193.9788208, 208.4597931, -274.8843079, 280.7315369, -474.7103577, 483.3440857
2: -282.0811462, 228.5283966, -398.1069336, 307.1251831, -589.2062988, 626.6353149
3: -114.9791260, 281.8475647, -156.3122864, 391.6330872, -506.6121521, 438.1598511
4: -314.5017090, 226.3350677, -443.3058777, 303.6446228, -618.1463013, 669.6409302

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -343.6112366, 288.9629822, -343.6112366, 288.9629822, -632.5742188, 632.5742188
1: -274.8843079, 280.7315369, -274.8843079, 280.7315369, -555.6158447, 555.6158447
2: -398.1069336, 307.1251831, -398.1069336, 307.1251831, -705.2321167, 705.2321167
3: -156.3122864, 391.6330872, -156.3122864, 391.6330872, -547.9453735, 547.9453735
4: -443.3058777, 303.6446228, -443.3058777, 303.6446228, -746.9505005, 746.9505005

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -243.1517029, 214.5575256, -277.1587219, 244.9257812, -488.0774841, 491.7162476
1: -193.9788208, 208.4597931, -221.0946198, 237.6749420, -431.6537476, 429.5543823
2: -282.0811462, 228.5283966, -321.5402527, 260.5017395, -542.5828857, 550.0685425
3: -114.9791260, 281.8475647, -132.1236572, 320.6901550, -435.6692200, 413.9712219
4: -314.5017090, 226.3350677, -359.0209351, 258.1718445, -572.6735229, 585.3558960

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -343.6112366, 288.9629822, -277.1587219, 244.9257812, -588.5369873, 566.1217041
1: -274.8843079, 280.7315369, -221.0946198, 237.6749420, -512.5592651, 501.8261719
2: -398.1069336, 307.1251831, -321.5402527, 260.5017395, -658.6086426, 628.6652832
3: -156.3122864, 391.6330872, -132.1236572, 320.6901550, -477.0024414, 523.7567139
4: -443.3058777, 303.6446228, -359.0209351, 258.1718445, -701.4777222, 662.6655273

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -243.1517029, 214.5575256, -399.4542847, 334.7368469, -577.8885498, 614.0118408
1: -193.9788208, 208.4597931, -319.7079773, 324.8833008, -518.8620605, 528.1677246
2: -282.0811462, 228.5283966, -463.7905884, 355.3358459, -637.4169922, 692.3189697
3: -114.9791260, 281.8475647, -181.5567322, 455.0501709, -570.0292969, 463.4042664
4: -314.5017090, 226.3350677, -517.1308594, 351.6369629, -666.1386719, 743.4658813

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -343.6112366, 288.9629822, -399.4542847, 334.7368469, -678.3480835, 688.4172363
1: -274.8843079, 280.7315369, -319.7079773, 324.8833008, -599.7675781, 600.4393921
2: -398.1069336, 307.1251831, -463.7905884, 355.3358459, -753.4427490, 770.9157715
3: -156.3122864, 391.6330872, -181.5567322, 455.0501709, -611.3623657, 573.1898193
4: -443.3058777, 303.6446228, -517.1308594, 351.6369629, -794.9428711, 820.7755127

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -277.1587219, 244.9257812, -243.1517029, 214.5575256, -491.7162476, 488.0774841
1: -221.0946198, 237.6749420, -193.9788208, 208.4597931, -429.5543823, 431.6537476
2: -321.5402527, 260.5017395, -282.0811462, 228.5283966, -550.0685425, 542.5828857
3: -132.1236572, 320.6901550, -114.9791260, 281.8475647, -413.9712219, 435.6692200
4: -359.0209351, 258.1718445, -314.5017090, 226.3350677, -585.3558350, 572.6735229

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -277.1587219, 244.9257812, -343.6112366, 288.9629822, -566.1217041, 588.5369873
1: -221.0946198, 237.6749420, -274.8843079, 280.7315369, -501.8261414, 512.5592651
2: -321.5402527, 260.5017395, -398.1069336, 307.1251831, -628.6652832, 658.6086426
3: -132.1236572, 320.6901550, -156.3122864, 391.6330872, -523.7567139, 477.0024414
4: -359.0209351, 258.1718445, -443.3058777, 303.6446228, -662.6655273, 701.4776611

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -399.4542847, 334.7368469, -243.1517029, 214.5575256, -614.0118408, 577.8885498
1: -319.7079773, 324.8833008, -193.9788208, 208.4597931, -528.1677856, 518.8621216
2: -463.7905884, 355.3358459, -282.0811462, 228.5283966, -692.3189697, 637.4169922
3: -181.5567322, 455.0501709, -114.9791260, 281.8475647, -463.4042358, 570.0292969
4: -517.1308594, 351.6369629, -314.5017090, 226.3350677, -743.4658813, 666.1386719

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -399.4542847, 334.7368469, -343.6112366, 288.9629822, -688.4172363, 678.3480835
1: -319.7079773, 324.8833008, -274.8843079, 280.7315369, -600.4393311, 599.7675781
2: -463.7905884, 355.3358459, -398.1069336, 307.1251831, -770.9157715, 753.4427490
3: -181.5567322, 455.0501709, -156.3122864, 391.6330872, -573.1898193, 611.3623657
4: -517.1308594, 351.6369629, -443.3058777, 303.6446228, -820.7755127, 794.9428711

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -277.1587219, 244.9257812, -277.1587219, 244.9257812, -522.0844727, 522.0844727
1: -221.0946198, 237.6749420, -221.0946198, 237.6749420, -458.7695618, 458.7695618
2: -321.5402527, 260.5017395, -321.5402527, 260.5017395, -582.0419312, 582.0419312
3: -132.1236572, 320.6901550, -132.1236572, 320.6901550, -452.8138123, 452.8138123
4: -359.0209351, 258.1718445, -359.0209351, 258.1718445, -617.1926880, 617.1926270

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -277.1587219, 244.9257812, -399.4542847, 334.7368469, -611.8955688, 644.3800659
1: -221.0946198, 237.6749420, -319.7079773, 324.8833008, -545.9779053, 557.3828125
2: -321.5402527, 260.5017395, -463.7905884, 355.3358459, -676.8759766, 724.2923584
3: -132.1236572, 320.6901550, -181.5567322, 455.0501709, -587.1737671, 502.2468262
4: -359.0209351, 258.1718445, -517.1308594, 351.6369629, -710.6578979, 775.3027344

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -399.4542847, 334.7368469, -277.1587219, 244.9257812, -644.3800659, 611.8955688
1: -319.7079773, 324.8833008, -221.0946198, 237.6749420, -557.3828735, 545.9779053
2: -463.7905884, 355.3358459, -321.5402527, 260.5017395, -724.2923584, 676.8759155
3: -181.5567322, 455.0501709, -132.1236572, 320.6901550, -502.2468262, 587.1737061
4: -517.1308594, 351.6369629, -359.0209351, 258.1718445, -775.3027344, 710.6578979

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -399.4542847, 334.7368469, -399.4542847, 334.7368469, -734.1911621, 734.1911621
1: -319.7079773, 324.8833008, -319.7079773, 324.8833008, -644.5910645, 644.5910645
2: -463.7905884, 355.3358459, -463.7905884, 355.3358459, -819.1264648, 819.1264648
3: -181.5567322, 455.0501709, -181.5567322, 455.0501709, -636.6068726, 636.6068726
4: -517.1308594, 351.6369629, -517.1308594, 351.6369629, -868.7678223, 868.7677612

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.28 + 286.72 = 288.99 seconds
