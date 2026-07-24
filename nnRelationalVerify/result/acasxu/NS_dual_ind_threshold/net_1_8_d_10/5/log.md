## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 157.33074007686602


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-101.0967865, 86.4863739, -101.0967865, 86.4863739, -187.5831146, 187.5831299)
1: (-378.3140564, 322.5913086, -378.3140564, 322.5913086, -700.9053955, 700.9053955)
2: (-209.0144958, 331.6951904, -209.0144958, 331.6951904, -540.7097168, 540.7097168)
3: (-347.8090515, 296.2051392, -347.8090515, 296.2051392, -644.0141602, 644.0141602)
4: (-257.1677856, 333.0890503, -257.1677856, 333.0890503, -590.2568359, 590.2568359)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.11 + 2.42 = 3.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -157.3323134, upper bound: 157.3323134

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3320955, upper bound: 157.3321766
time: 1.24 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3321687, upper bound: 157.3321687
time: 0.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.28 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.28
Output dim: 0, lower bound: -157.3320955, upper bound: 157.3321766
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.28
Output dim: 0, lower bound: -157.3321687, upper bound: 157.3321687

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -95.8630753, 82.0812988, -98.1439590, 83.9541779, -179.8172607, 180.2252350
1: -359.0657654, 306.1313477, -367.3385925, 313.3635254, -672.4291992, 673.4699097
2: -197.8675995, 313.4418335, -202.7401428, 321.4366455, -519.3042603, 516.1820068
3: -329.5890808, 281.3823547, -337.5264587, 287.7499390, -617.3389282, 618.9088135
4: -243.9252625, 315.3025513, -249.7127075, 322.9081726, -566.8334351, 565.0151367

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3320793, upper bound: 157.3321389
time: 0.98 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3320757, upper bound: 157.3321389
time: 1.04 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -106.5379486, 91.2489395, -98.4418335, 84.2735367, -190.8114319, 189.6907349
1: -399.5735474, 341.2315674, -368.2709045, 314.4390259, -714.0125732, 709.5024414
2: -219.9133911, 348.9153137, -203.6147308, 323.3686218, -543.2819824, 552.5299072
3: -366.6865845, 313.2791443, -338.5883484, 288.7008972, -655.3874512, 651.8673706
4: -270.8165588, 350.6130676, -250.3196259, 324.6997375, -595.5162964, 600.9326782

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3321574, upper bound: 157.3321381
time: 1.03 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3321329, upper bound: 157.3321329
time: 1.00 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.38 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -157.3320793, upper bound: 157.3321389
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -157.3320757, upper bound: 157.3321389
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -157.3321574, upper bound: 157.3321381
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -157.3321329, upper bound: 157.3321329

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -95.2618256, 81.5966873, -96.9788208, 83.0066071, -178.2684326, 178.5754700
1: -356.7752991, 304.3562012, -362.9037170, 309.8874817, -666.6627808, 667.2597046
2: -196.6774902, 311.5847778, -200.4136658, 317.7979736, -514.4754028, 511.9984436
3: -327.4916992, 279.7485046, -333.4664917, 284.5315247, -612.0231934, 613.2149658
4: -242.3927460, 313.4628601, -246.7398376, 319.2421265, -561.6348877, 560.2026367

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3314978, upper bound: 157.3318368
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3318509, upper bound: 157.3319890
time: 2.10 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -94.2487946, 80.6834869, -97.8469849, 83.7642899, -178.0130463, 178.5304413
1: -352.9393921, 300.9978638, -366.3111572, 312.7910461, -665.7304688, 667.3089600
2: -194.6229553, 308.2867737, -202.2736359, 320.8651733, -515.4881592, 510.5604248
3: -324.0142517, 276.5958557, -336.4404602, 287.3083801, -611.3226318, 613.0363159
4: -239.8236847, 309.9900818, -248.8961487, 322.2827454, -562.1064453, 558.8861694

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3314946, upper bound: 157.3318527
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3318251, upper bound: 157.3319839
time: 1.11 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -105.7915878, 90.6292572, -97.0335922, 83.1084061, -188.8999634, 187.6628418
1: -396.7496948, 338.9768372, -362.9288330, 310.1915588, -706.9412231, 701.9055786
2: -218.3841400, 346.5251160, -200.7592468, 318.9102478, -537.2943726, 547.2843628
3: -364.0984497, 311.1818542, -333.6966858, 284.7585754, -648.8569946, 644.8782959
4: -268.9206238, 348.2086792, -246.7321472, 320.2121582, -589.1327515, 594.9407349

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3299821, upper bound: 157.3302564
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3320290, upper bound: 157.3320225
time: 1.04 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -104.9587402, 89.9004822, -98.1449738, 84.0820618, -189.0407562, 188.0454559
1: -393.5834656, 336.2124329, -367.2088318, 313.8453369, -707.4287109, 703.4212036
2: -216.7348633, 343.8597107, -203.1500397, 322.7901917, -539.5250244, 547.0096436
3: -361.2262573, 308.6534424, -337.4904175, 288.2394104, -649.4656982, 646.1438599
4: -266.8009949, 345.4918213, -249.4894714, 324.0650330, -590.8660278, 594.9812622

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3299813, upper bound: 157.3302563
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3320073, upper bound: 157.3320073
time: 1.10 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.23 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -157.3314978, upper bound: 157.3318368
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -157.3318509, upper bound: 157.3319890
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -157.3314946, upper bound: 157.3318527
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -157.3318251, upper bound: 157.3319839
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -157.3299821, upper bound: 157.3302564
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -157.3320290, upper bound: 157.3320225
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -157.3299813, upper bound: 157.3302563
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -157.3320073, upper bound: 157.3320073

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -85.6869354, 73.5945206, -93.0911865, 79.8313828, -165.5183105, 166.6856995
1: -321.1247864, 274.8855286, -348.6851807, 297.8549500, -618.9796143, 623.5704956
2: -177.0827789, 281.9274597, -192.4223022, 305.4389343, -482.5217285, 474.3496399
3: -294.6037903, 252.7151642, -320.1787109, 273.5704346, -568.1741943, 572.8938599
4: -217.6674652, 283.4854736, -236.6735992, 306.8835449, -524.5509644, 520.1589966

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3314978, upper bound: 157.3318180
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3314978, upper bound: 157.3318368
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -92.5925140, 79.3342972, -95.3928070, 81.6666946, -174.2592163, 174.7271118
1: -346.5907593, 295.9483643, -356.9167175, 304.8879395, -651.4786987, 652.8651123
2: -191.2591248, 303.2070923, -197.1710510, 312.8170166, -504.0761414, 500.3781433
3: -318.2990112, 272.0475769, -328.0172119, 279.9468994, -598.2459106, 600.0647583
4: -235.5311279, 304.9696960, -242.6495972, 314.1912231, -549.7222900, 547.6192627

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3318509, upper bound: 157.3318724
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3318509, upper bound: 157.3319890
time: 1.65 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -84.7546616, 72.7754059, -94.2834244, 80.8547745, -165.6094360, 167.0588379
1: -317.5723267, 271.8207092, -353.2713013, 301.8122253, -619.3843384, 625.0919800
2: -175.1936340, 278.8532410, -194.9511108, 309.5929871, -484.7866211, 473.8043213
3: -291.3792114, 249.9004974, -324.2704163, 277.2698975, -568.6491089, 574.1708374
4: -215.3040009, 280.3792114, -239.6699524, 310.9588318, -526.2627563, 520.0491333

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3314946, upper bound: 157.3318155
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3314946, upper bound: 157.3318528
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -91.7665482, 78.5830688, -96.2583847, 82.4177475, -174.1842957, 174.8414612
1: -343.4643250, 293.2062683, -360.2906799, 307.7843018, -651.2485962, 653.4968872
2: -189.5997620, 300.5509338, -199.0403137, 315.8732605, -505.4730225, 499.5912476
3: -315.4698181, 269.4594727, -330.9803467, 282.7166748, -598.1865234, 600.4396973
4: -233.4282227, 302.1276855, -244.8040314, 317.2157593, -550.6439819, 546.9316406

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3318251, upper bound: 157.3318392
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3318251, upper bound: 157.3319839
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -102.9131393, 88.1790390, -95.6417847, 81.9202805, -184.8334045, 183.8208313
1: -386.0767822, 329.9278870, -357.7240295, 305.7697449, -691.8463745, 687.6519165
2: -212.5333710, 337.4399719, -197.9303131, 314.5599976, -527.0933228, 535.3703003
3: -354.2684631, 302.8996277, -328.9254456, 280.7101746, -634.9786377, 631.8250732
4: -261.5375671, 339.0027771, -243.1426086, 315.7858582, -577.3234253, 582.1453857

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3320154, upper bound: 157.3318615
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3320154, upper bound: 157.3319859
time: 1.21 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -102.1612701, 87.5192795, -96.6572342, 82.8116226, -184.9729004, 184.1764832
1: -383.2067871, 327.4226074, -361.6557312, 309.1295471, -692.3363037, 689.0783691
2: -211.0487518, 335.0538330, -200.1342010, 318.1396790, -529.1884155, 535.1880493
3: -351.6745605, 300.6137390, -332.3932495, 283.9189148, -635.5935059, 633.0069580
4: -259.6180115, 336.5670471, -245.6580963, 319.3298340, -578.9478760, 582.2251587

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3319839, upper bound: 157.3318251
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -157.3319839, upper bound: 157.3318250
time: 1.07 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.15 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -157.3314978, upper bound: 157.3318180
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -157.3314978, upper bound: 157.3318368
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -157.3318509, upper bound: 157.3318724
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -157.3318509, upper bound: 157.3319890
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -157.3314946, upper bound: 157.3318155
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -157.3314946, upper bound: 157.3318528
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -157.3318251, upper bound: 157.3318392
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -157.3318251, upper bound: 157.3319839
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -157.3320154, upper bound: 157.3318615
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -157.3320154, upper bound: 157.3319859
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -157.3319839, upper bound: 157.3318251
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -157.3319839, upper bound: 157.3318250

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -85.6869354, 73.5945206, -91.0516434, 78.1499634, -163.8368530, 164.6461639
1: -321.1247864, 274.8855286, -341.3738403, 291.3783875, -612.5031128, 616.2592163
2: -177.0827789, 281.9274597, -188.0112915, 298.1535950, -475.2363586, 469.9386597
3: -294.6037903, 252.7151642, -313.0956116, 268.0051880, -562.6090088, 565.8107300
4: -217.6674652, 283.4854736, -231.5035706, 300.2221680, -517.8895874, 514.9890137

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3307134, upper bound: 157.3306165
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3305630, upper bound: 157.3306136
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -85.6869354, 73.5945206, -102.2545624, 87.6908112, -173.3777313, 175.8490601
1: -321.1247864, 274.8855286, -383.7306519, 328.0180969, -649.1427612, 658.6160278
2: -177.0827789, 281.9274597, -211.0969391, 335.2699890, -512.3527832, 493.0243530
3: -294.6037903, 252.7151642, -352.0261536, 301.1278687, -595.7316284, 604.7412720
4: -217.6674652, 283.4854736, -259.8071899, 336.8755493, -554.5430298, 543.2926636

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3307134, upper bound: 157.3306165
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3305630, upper bound: 157.3306137
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -92.5925140, 79.3342972, -93.1697006, 79.8420486, -172.4345703, 172.5039978
1: -346.5907593, 295.9483643, -348.7831726, 297.8517456, -644.4425049, 644.7315063
2: -191.2591248, 303.2070923, -192.4420319, 305.0295105, -496.2886353, 495.6491089
3: -318.2990112, 272.0475769, -320.2582703, 273.7868042, -592.0858154, 592.3058472
4: -235.5311279, 304.9696960, -237.0289764, 306.8546143, -542.3856812, 541.9986572

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3305279, upper bound: 157.3302823
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3301535, upper bound: 157.3301630
time: 1.20 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -92.5925140, 79.3342972, -103.4861069, 88.6756058, -181.2681274, 182.8204041
1: -346.5907593, 295.9483643, -388.1774292, 331.7963867, -678.3871460, 684.1257935
2: -191.2591248, 303.2070923, -213.6813202, 339.1848755, -530.4439697, 516.8882446
3: -318.2990112, 272.0475769, -356.2023010, 304.5809021, -622.8798828, 628.2498779
4: -235.5311279, 304.9696960, -263.0263062, 340.7889404, -576.3200684, 567.9959717

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3305279, upper bound: 157.3302823
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3301535, upper bound: 157.3301630
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -84.7546616, 72.7754059, -92.0560989, 79.0218506, -163.7765045, 164.8314667
1: -317.5723267, 271.8207092, -345.2317810, 294.7337341, -612.3059692, 617.0524292
2: -175.1936340, 278.8532410, -190.1733704, 301.7650757, -476.9586792, 469.0265808
3: -291.3792114, 249.9004974, -316.5244751, 271.1829529, -562.5621338, 566.4249268
4: -215.3040009, 280.3792114, -234.0041656, 303.7388306, -519.0428467, 514.3833618

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3307134, upper bound: 157.3306165
time: 1.17 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3305346, upper bound: 157.3306136
time: 1.17 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -84.7546616, 72.7754059, -104.8920517, 89.8998413, -174.6544647, 177.6674347
1: -317.5723267, 271.8207092, -393.9315491, 336.3408813, -653.9131470, 665.7520752
2: -175.1936340, 278.8532410, -216.5397034, 343.7680359, -518.9616699, 495.3929443
3: -291.3792114, 249.9004974, -361.2118530, 308.8996277, -600.2788086, 611.1123657
4: -215.3040009, 280.3792114, -266.4989929, 345.3492126, -560.6531372, 546.8781738

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3307134, upper bound: 157.3306165
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3305346, upper bound: 157.3306137
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -91.7665482, 78.5830688, -93.8258438, 80.4198914, -172.1864319, 172.4089050
1: -343.4643250, 293.2062683, -351.3781738, 300.1170044, -643.5812988, 644.5843506
2: -189.5997620, 300.5509338, -193.8897400, 307.4498901, -497.0496216, 494.4406433
3: -315.4698181, 269.4594727, -322.4959106, 275.9752502, -591.4450684, 591.9553223
4: -233.4282227, 302.1276855, -238.6428070, 309.2255554, -542.6538086, 540.7704468

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3304334, upper bound: 157.3301685
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3301007, upper bound: 157.3300930
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -91.7665482, 78.5830688, -105.7471008, 90.5657654, -182.3323059, 184.3301697
1: -343.4643250, 293.2062683, -396.9617920, 338.9230042, -682.3873291, 690.1679688
2: -189.5997620, 300.5509338, -218.3578491, 346.4274597, -536.0272217, 518.9087524
3: -315.4698181, 269.4594727, -364.1086121, 311.2647705, -626.7346191, 633.5679321
4: -233.4282227, 302.1276855, -268.7636719, 348.0231323, -581.4513550, 570.8913574

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3303308, upper bound: 157.3301685
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3301007, upper bound: 157.3300930
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -102.9131393, 88.1790390, -93.1697006, 79.8420486, -182.7551880, 181.3487396
1: -386.0767822, 329.9278870, -348.7831726, 297.8517456, -683.9284058, 678.7110596
2: -212.5333710, 337.4399719, -192.4420319, 305.0295105, -517.5628662, 529.8820190
3: -354.2684631, 302.8996277, -320.2582703, 273.7868042, -628.0552979, 623.1578369
4: -261.5375671, 339.0027771, -237.0289764, 306.8546143, -568.3922119, 576.0317383

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3305381, upper bound: 157.3302847
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3301349, upper bound: 157.3301520
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -102.9131393, 88.1790390, -103.4861069, 88.6756058, -191.5887451, 191.6651459
1: -386.0767822, 329.9278870, -388.1774292, 331.7963867, -717.8731689, 718.1053467
2: -212.5333710, 337.4399719, -213.6813202, 339.1848755, -551.7182617, 551.1212769
3: -354.2684631, 302.8996277, -356.2023010, 304.5809021, -658.8493042, 659.1019287
4: -261.5375671, 339.0027771, -263.0263062, 340.7889404, -602.3265381, 602.0290527

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3305381, upper bound: 157.3302846
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3301349, upper bound: 157.3301520
time: 1.03 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -102.1612701, 87.5192795, -93.8258438, 80.4198914, -182.5811615, 181.3451233
1: -383.2067871, 327.4226074, -351.3781738, 300.1170044, -683.3237915, 678.8007812
2: -211.0487518, 335.0538330, -193.8897400, 307.4498901, -518.4986572, 528.9436035
3: -351.6745605, 300.6137390, -322.4959106, 275.9752502, -627.6497803, 623.1096191
4: -259.6180115, 336.5670471, -238.6428070, 309.2255554, -568.8435669, 575.2098389

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3304362, upper bound: 157.3301688
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3300840, upper bound: 157.3300840
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -102.1612701, 87.5192795, -105.7471008, 90.5657654, -192.7270355, 193.2663574
1: -383.2067871, 327.4226074, -396.9617920, 338.9230042, -722.1297607, 724.3843994
2: -211.0487518, 335.0538330, -218.3578491, 346.4274597, -557.4761963, 553.4116821
3: -351.6745605, 300.6137390, -364.1086121, 311.2647705, -662.9393311, 664.7223511
4: -259.6180115, 336.5670471, -268.7636719, 348.0231323, -607.6411133, 605.3306885

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3304362, upper bound: 157.3301688
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -157.3300840, upper bound: 157.3300840
time: 0.91 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.84 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3307134, upper bound: 157.3306165
NS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3305630, upper bound: 157.3306136
NS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3307134, upper bound: 157.3306165
NS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3305630, upper bound: 157.3306137
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3305279, upper bound: 157.3302823
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3301535, upper bound: 157.3301630
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3305279, upper bound: 157.3302823
NS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3301535, upper bound: 157.3301630
NS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3307134, upper bound: 157.3306165
NS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3305346, upper bound: 157.3306136
NS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3307134, upper bound: 157.3306165
NS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3305346, upper bound: 157.3306137
NS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3304334, upper bound: 157.3301685
NS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3301007, upper bound: 157.3300930
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3303308, upper bound: 157.3301685
NS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3301007, upper bound: 157.3300930
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3305381, upper bound: 157.3302847
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3301349, upper bound: 157.3301520
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3305381, upper bound: 157.3302846
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3301349, upper bound: 157.3301520
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3304362, upper bound: 157.3301688
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3300840, upper bound: 157.3300840
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3304362, upper bound: 157.3301688
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -157.3300840, upper bound: 157.3300840

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 3.53 + 82.11 = 85.64 seconds
