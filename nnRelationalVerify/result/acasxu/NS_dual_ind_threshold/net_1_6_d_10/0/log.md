## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 380.96572808527804


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-120.7598648, 310.2168884, -120.7598648, 310.2168884, -430.9767456, 430.9767456)
1: (-306.0167542, 469.4156189, -306.0167542, 469.4156189, -775.4323730, 775.4323730)
2: (-197.4686890, 458.7374573, -197.4686890, 458.7374573, -656.2061768, 656.2061768)
3: (-330.1389465, 542.4745483, -330.1389465, 542.4745483, -872.6135254, 872.6134644)
4: (-287.0395203, 524.3436890, -287.0395203, 524.3436890, -811.3831177, 811.3831177)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.82 + 1.96 = 2.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -380.9771574, upper bound: 380.9771574

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9771470, upper bound: 380.9766991
time: 1.15 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9766991, upper bound: 380.9766991
time: 0.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.83 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.83
Output dim: 0, lower bound: -380.9771470, upper bound: 380.9766991
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.83
Output dim: 0, lower bound: -380.9766991, upper bound: 380.9766991

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -102.1709671, 261.3860168, -114.3362198, 294.5241699, -396.6950684, 375.7222290
1: -258.5334167, 395.5080872, -290.0555725, 446.3215942, -704.8548584, 685.5635986
2: -166.9586182, 385.7630310, -186.8040314, 435.1654968, -602.1241455, 572.5670776
3: -278.8269348, 456.4924622, -312.8416443, 515.4390869, -794.2659912, 769.3339844
4: -243.0326538, 440.8841858, -271.7865601, 498.0593872, -741.0920410, 712.6705933

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9638806, upper bound: 380.9643346
time: 0.67 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9623035, upper bound: 380.9634022
time: 0.61 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -118.8791046, 305.5854492, -119.3134308, 306.6369019, -425.5159912, 424.8988647
1: -301.2553406, 462.6041870, -302.3125000, 464.1497498, -765.4050293, 764.9166870
2: -194.1865692, 452.0354309, -194.8987427, 453.5514832, -647.7380371, 646.9342041
3: -325.1773376, 534.5447998, -326.2976990, 536.3432617, -861.5205078, 860.8423462
4: -282.2735596, 516.6941528, -283.3926697, 518.4217529, -800.6953125, 800.0867920

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9688829, upper bound: 380.9686109
time: 0.65 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9623035, upper bound: 380.9670411
time: 0.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.18 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.18
Output dim: 0, lower bound: -380.9638806, upper bound: 380.9643346
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.18
Output dim: 0, lower bound: -380.9623035, upper bound: 380.9634022
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -380.9688829, upper bound: 380.9686109
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -380.9623035, upper bound: 380.9670411

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -116.4445114, 299.3266907, -115.4604187, 296.7470703, -413.1915894, 414.7871094
1: -294.8954773, 453.2035522, -292.2443542, 449.3006287, -744.1961060, 745.4478760
2: -190.0927124, 442.5434265, -188.4377594, 438.5581665, -628.6508179, 630.9810791
3: -318.5371399, 523.6215820, -315.7623901, 519.0971069, -837.6341553, 839.3839722
4: -276.6002197, 506.0847168, -274.4066162, 501.6541748, -778.2543945, 780.4913330

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9655942, upper bound: 380.9649652
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9672541, upper bound: 380.9659970
time: 0.81 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -114.7860565, 294.0978699, -133.7529602, 340.8833008, -455.6693726, 427.8507996
1: -289.4912109, 445.7116089, -336.0966187, 516.3527222, -805.8439331, 781.8081055
2: -187.1117554, 434.8902283, -217.6470032, 502.3092957, -689.4210205, 652.5372314
3: -313.7854919, 515.0421753, -365.0690918, 597.5002441, -911.2856445, 880.1111450
4: -272.7771606, 496.7930908, -318.3937683, 574.4002075, -847.1773682, 815.1868286

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9632818, upper bound: 380.9641155
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9652623, upper bound: 380.9652623
time: 0.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.10 seconds
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.10
Output dim: 0, lower bound: -380.9655942, upper bound: 380.9649652
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -380.9672541, upper bound: 380.9659970
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.10
Output dim: 0, lower bound: -380.9632818, upper bound: 380.9641155
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.10
Output dim: 0, lower bound: -380.9652623, upper bound: 380.9652623

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -113.2716904, 291.7255554, -113.5330505, 292.1243286, -405.3960266, 405.2586060
1: -287.0264282, 441.8933105, -287.4485779, 442.4291687, -729.4555664, 729.3417969
2: -184.9462128, 431.3915405, -185.2638702, 431.7685852, -616.7147827, 616.6553955
3: -310.0375366, 510.5404358, -310.5914001, 511.1463013, -821.1837158, 821.1316528
4: -268.9721985, 493.4302368, -269.7741089, 493.9593811, -762.9315186, 763.2043457

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9651701, upper bound: 380.9638768
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9672541, upper bound: 380.9659970
time: 1.18 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.68 seconds
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -380.9651701, upper bound: 380.9638768
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -380.9672541, upper bound: 380.9659970

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -112.4986725, 289.7720642, -112.4619904, 289.4015198, -401.9002075, 402.2340698
1: -285.0617065, 439.0145264, -284.7319031, 438.4011230, -723.4628296, 723.7462769
2: -183.6560974, 428.5306702, -183.4857483, 427.7785645, -611.4346924, 612.0164185
3: -307.9367371, 507.1999207, -307.6806641, 506.4842529, -814.4210205, 814.8805542
4: -267.1188965, 490.1815491, -267.2108459, 489.4280701, -756.5469971, 757.3923340

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9628674, upper bound: 380.9627077
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9672541, upper bound: 380.9659970
time: 0.87 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.35 seconds
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.35
Output dim: 0, lower bound: -380.9628674, upper bound: 380.9627077
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.35
Output dim: 0, lower bound: -380.9672541, upper bound: 380.9659970

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -111.3981018, 286.9939575, -111.9047928, 287.9855957, -399.3836975, 398.8987427
1: -282.3124695, 434.8010559, -283.3146667, 436.2551270, -718.5676270, 718.1156616
2: -181.8485718, 424.4610291, -182.5737610, 425.6957703, -607.5443115, 607.0346680
3: -304.8905029, 502.3545837, -306.1287842, 504.0123901, -808.9028931, 808.4833374
4: -264.4818726, 485.5103149, -265.8785706, 487.0349121, -751.5166626, 751.3889160

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9666425, upper bound: 380.9659014
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9670028, upper bound: 380.9659970
time: 0.91 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.39 seconds
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.39
Output dim: 0, lower bound: -380.9666425, upper bound: 380.9659014
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.39
Output dim: 0, lower bound: -380.9670028, upper bound: 380.9659970

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -109.1394806, 281.2298889, -108.9491272, 280.1452942, -389.2847900, 390.1790161
1: -276.7825623, 426.0996399, -277.3763123, 424.4076843, -701.1901855, 703.4759521
2: -178.2467346, 416.1659851, -178.6744690, 414.2894897, -592.5362549, 594.8403931
3: -298.6938782, 492.3502808, -298.2582703, 490.9793701, -789.6732178, 790.6085205
4: -258.9990845, 476.0036926, -258.5288696, 474.7673035, -733.7663574, 734.5324707

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9666425, upper bound: 380.9659014
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9666425, upper bound: 380.9659014
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -110.5343094, 284.7401733, -110.4797821, 284.2623596, -394.7966614, 395.2199402
1: -280.0118103, 431.4539490, -279.5233459, 430.7288208, -710.7406006, 710.9772339
2: -180.3761292, 421.0684509, -180.1308441, 420.0856018, -600.4617310, 601.1990967
3: -302.5249023, 498.4201050, -302.2250977, 497.5073242, -800.0321045, 800.6452026
4: -262.4782104, 481.6699219, -262.5693054, 480.6873779, -743.1655884, 744.2392578

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9670028, upper bound: 380.9659970
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9670028, upper bound: 380.9659970
time: 1.05 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.63 seconds
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -380.9666425, upper bound: 380.9659014
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -380.9666425, upper bound: 380.9659014
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -380.9670028, upper bound: 380.9659970
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -380.9670028, upper bound: 380.9659970

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -107.6499252, 277.4054871, -108.9491272, 280.1452942, -387.7951660, 386.3545837
1: -272.8387451, 420.3706970, -277.3763123, 424.4076843, -697.2464600, 697.7470093
2: -175.7212982, 410.3366394, -178.6744690, 414.2894897, -590.0108032, 589.0111084
3: -294.6065369, 485.6777954, -298.2582703, 490.9793701, -785.5859375, 783.9360352
4: -255.5363770, 469.5085144, -258.5288696, 474.7673035, -730.3035278, 728.0373535

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663672, upper bound: 380.9658418
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9660596, upper bound: 380.9657476
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9629596, upper bound: 380.9641289
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9664550, upper bound: 380.9651052
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -124.6903076, 318.3165894, -108.9491272, 280.1452942, -404.8356018, 427.2657166
1: -312.8399658, 482.5302124, -277.3763123, 424.4076843, -737.2476196, 759.7819214
2: -202.4766388, 469.2543945, -178.6744690, 414.2894897, -616.7661133, 647.9288330
3: -340.1546936, 558.3353882, -298.2582703, 490.9793701, -830.3115845, 855.8569946
4: -296.7083435, 536.5305786, -258.5288696, 474.7673035, -771.4754639, 795.0593872

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663672, upper bound: 380.9658418
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9660596, upper bound: 380.9657476
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9629596, upper bound: 380.9641289
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9664550, upper bound: 380.9651052
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -109.0328903, 280.8949585, -110.4797821, 284.2623596, -393.2952271, 391.3747253
1: -276.0534363, 425.6777039, -279.5233459, 430.7288208, -706.7822266, 705.2010498
2: -177.8319244, 415.2210999, -180.1308441, 420.0856018, -597.9175415, 595.3517456
3: -298.3986511, 491.7099304, -302.2250977, 497.5073242, -795.9059448, 793.9350586
4: -258.9868164, 475.1401062, -262.5693054, 480.6873779, -739.6741333, 737.7092896

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9649450, upper bound: 380.9635083
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9649450, upper bound: 380.9659968
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -126.4421921, 322.8362732, -110.4797821, 284.2623596, -410.7045593, 433.3160400
1: -317.1414185, 489.3787231, -279.5233459, 430.7288208, -747.8702393, 768.9020996
2: -205.2606354, 475.6784668, -180.1308441, 420.0856018, -625.3462524, 655.8091431
3: -345.0083618, 566.1856689, -302.2250977, 497.5073242, -841.8560181, 867.5738525
4: -300.9461975, 543.9959106, -262.5693054, 480.6873779, -781.6335449, 806.5651855

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9649450, upper bound: 380.9635083
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9649450, upper bound: 380.9659968
time: 0.66 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 3.76 seconds
NS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.76
Output dim: 0, lower bound: -380.9629596, upper bound: 380.9641289
NS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.76
Output dim: 0, lower bound: -380.9664550, upper bound: 380.9651052
NS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.76
Output dim: 0, lower bound: -380.9629596, upper bound: 380.9641289
NS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.76
Output dim: 0, lower bound: -380.9664550, upper bound: 380.9651052
NS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.76
Output dim: 0, lower bound: -380.9649450, upper bound: 380.9635083
NS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.76
Output dim: 0, lower bound: -380.9649450, upper bound: 380.9659968
NS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.76
Output dim: 0, lower bound: -380.9649450, upper bound: 380.9635083
NS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.76
Output dim: 0, lower bound: -380.9649450, upper bound: 380.9659968

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -107.4432831, 276.8882446, -108.5963898, 279.2680359, -386.7112732, 385.4845886
1: -272.3182373, 419.5860291, -276.5018616, 423.0708618, -695.3890381, 696.0878906
2: -175.3843536, 409.5687866, -178.0997162, 412.9897156, -588.3740234, 587.6685181
3: -294.0517273, 484.7684021, -297.3167419, 489.4352722, -783.4868774, 782.0851440
4: -255.0421295, 468.6386108, -257.6817017, 473.3031006, -728.3452148, 726.3201904

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9697723, upper bound: 380.9697723
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9697723, upper bound: 380.9698159
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -124.5003738, 317.8476562, -108.5963898, 279.2680359, -403.7684021, 426.4440308
1: -312.3764954, 481.8274841, -276.5018616, 423.0708618, -735.4473877, 758.1550293
2: -202.1714630, 468.5631104, -178.0997162, 412.9897156, -615.1611938, 646.6628418
3: -339.6547852, 557.5219116, -297.3167419, 489.4352722, -828.1636353, 854.0477905
4: -296.2527161, 535.7510986, -257.6817017, 473.3031006, -769.5557251, 793.4328003

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9664550, upper bound: 380.9651006
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9664550, upper bound: 380.9651052
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -109.0328903, 280.8949585, -110.0667038, 283.2132568, -392.2461243, 390.9616394
1: -276.0534363, 425.6777039, -278.4831848, 429.1341858, -705.1876221, 704.1608887
2: -177.8319244, 415.2210999, -179.4550629, 418.5389099, -596.3708496, 594.6759644
3: -298.3986511, 491.7099304, -301.0836792, 495.6702271, -794.0687866, 792.7935791
4: -258.9868164, 475.1401062, -261.5837097, 478.9149780, -737.9017944, 736.7237549

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9653276, upper bound: 380.9680319
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9653276, upper bound: 380.9696832
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -126.4421921, 322.8362732, -110.0667038, 283.2132568, -409.6554565, 432.9029541
1: -317.1414185, 489.3787231, -278.4831848, 429.1341858, -746.2756348, 767.8619385
2: -205.2606354, 475.6784668, -179.4550629, 418.5389099, -623.7995605, 655.1333618
3: -345.0083618, 566.1856689, -301.0836792, 495.6702271, -839.8950195, 866.4645996
4: -300.9461975, 543.9959106, -261.5837097, 478.9149780, -779.8612061, 805.5795898

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9630149, upper bound: 380.9649042
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9648379, upper bound: 380.9649041
time: 0.71 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 3.33 seconds
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.33
Output dim: 0, lower bound: -380.9697723, upper bound: 380.9697723
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.33
Output dim: 0, lower bound: -380.9697723, upper bound: 380.9698159
NS_A2_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.33
Output dim: 0, lower bound: -380.9664550, upper bound: 380.9651006
NS_A2_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.33
Output dim: 0, lower bound: -380.9664550, upper bound: 380.9651052
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.33
Output dim: 0, lower bound: -380.9653276, upper bound: 380.9680319
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.33
Output dim: 0, lower bound: -380.9653276, upper bound: 380.9696832
NS_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -380.9630149, upper bound: 380.9649042
NS_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -380.9648379, upper bound: 380.9649041

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -106.9528122, 275.3466492, -108.5963898, 279.2680359, -386.2208557, 383.9430237
1: -272.4193115, 417.1882019, -276.5018616, 423.0708618, -695.4899902, 693.6900024
2: -175.4478912, 407.2334290, -178.0997162, 412.9897156, -588.4376221, 585.3330688
3: -292.9085999, 482.6208801, -297.3167419, 489.4352722, -782.3438721, 779.7713623
4: -253.6826630, 466.7431030, -257.6817017, 473.3031006, -726.9857788, 724.4248047

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9669374, upper bound: 380.9673325
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9689165, upper bound: 380.9689165
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -108.4261322, 279.3305664, -108.5963898, 279.2680359, -387.6941528, 387.9269409
1: -274.4685974, 423.3350525, -276.5018616, 423.0708618, -697.5393677, 699.8368530
2: -176.8138123, 412.8789368, -178.0997162, 412.9897156, -589.8035278, 590.9786377
3: -296.7483215, 488.9632263, -297.3167419, 489.4352722, -786.1020508, 786.2799683
4: -257.5596313, 472.4891968, -257.6817017, 473.3031006, -730.8627319, 730.1708984

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9669374, upper bound: 380.9676231
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9689165, upper bound: 380.9689656
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -125.7538757, 320.5909729, -108.5963898, 279.2680359, -405.0219116, 429.1873474
1: -317.3356934, 485.4917603, -276.5018616, 423.0708618, -740.4065552, 761.7288208
2: -205.1557312, 472.9804077, -178.0997162, 412.9897156, -618.1454468, 651.0800171
3: -343.4659119, 562.7498169, -297.3167419, 489.4352722, -831.9829102, 859.1081543
4: -298.4636230, 541.1392212, -257.6817017, 473.3031006, -771.7666626, 798.8209229

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9649150, upper bound: 380.9635181
time: 1.08 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9646673, upper bound: 380.9636175
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -125.7629089, 321.0937805, -108.5963898, 279.2680359, -405.0309143, 429.6901550
1: -315.3839417, 486.7881165, -276.5018616, 423.0708618, -738.4548340, 763.0696411
2: -204.1296692, 473.0707703, -178.0997162, 412.9897156, -617.1193848, 651.1703491
3: -343.1635742, 563.1499634, -297.3167419, 489.4352722, -831.5718384, 859.5682373
4: -299.3413391, 541.0545654, -257.6817017, 473.3031006, -772.6442871, 798.7362671

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9649150, upper bound: 380.9635203
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9646673, upper bound: 380.9636275
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -99.2833405, 255.4246826, -110.0667038, 283.2132568, -382.4965820, 365.4913635
1: -250.1900177, 386.4328003, -278.4831848, 429.1341858, -679.3242188, 664.9160156
2: -161.3197937, 377.0191345, -179.4550629, 418.5389099, -579.8587036, 556.4741821
3: -271.5458069, 445.9601746, -301.0836792, 495.6702271, -767.2158813, 747.0437012
4: -236.0625153, 430.9267883, -261.5837097, 478.9149780, -714.9774780, 692.5104370

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9671947, upper bound: 380.9678161
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9681571, upper bound: 380.9680319
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -108.6336746, 279.8709106, -110.0667038, 283.2132568, -391.8468933, 389.9375610
1: -275.0256348, 424.1694031, -278.4831848, 429.1341858, -704.1597900, 702.6525879
2: -177.1611023, 413.6959839, -179.4550629, 418.5389099, -595.7000122, 593.1509399
3: -297.3070068, 489.9585266, -301.0836792, 495.6702271, -792.9770508, 791.0421143
4: -258.0428162, 473.4209595, -261.5837097, 478.9149780, -736.9577637, 735.0045166

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9671947, upper bound: 380.9687179
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9681571, upper bound: 380.9686824
time: 0.71 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 2.37 seconds
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.37
Output dim: 0, lower bound: -380.9669374, upper bound: 380.9673325
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.37
Output dim: 0, lower bound: -380.9689165, upper bound: 380.9689165
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.37
Output dim: 0, lower bound: -380.9669374, upper bound: 380.9676231
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.37
Output dim: 0, lower bound: -380.9689165, upper bound: 380.9689656
NS_A2_B1_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 2.37
Output dim: 0, lower bound: -380.9649150, upper bound: 380.9635181
NS_A2_B1_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 2.37
Output dim: 0, lower bound: -380.9646673, upper bound: 380.9636175
NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 2.37
Output dim: 0, lower bound: -380.9649150, upper bound: 380.9635203
NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 2.37
Output dim: 0, lower bound: -380.9646673, upper bound: 380.9636275
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.37
Output dim: 0, lower bound: -380.9671947, upper bound: 380.9678161
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.37
Output dim: 0, lower bound: -380.9681571, upper bound: 380.9680319
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.37
Output dim: 0, lower bound: -380.9671947, upper bound: 380.9687179
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.37
Output dim: 0, lower bound: -380.9681571, upper bound: 380.9686824

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -103.9039993, 266.5040588, -110.3462219, 281.5076599, -385.4116211, 376.8502808
1: -264.1903992, 403.4578857, -280.0859375, 425.4723206, -689.6626587, 683.3521118
2: -170.5434723, 394.0783997, -181.0200806, 415.2046204, -585.7481079, 575.0985107
3: -284.4752197, 466.9515381, -301.7894287, 492.5526123, -776.8433228, 768.3408813
4: -246.6729889, 451.2542725, -262.3458252, 475.5311584, -722.2040405, 713.5999756

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9652307, upper bound: 380.9657366
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9652343, upper bound: 380.9651181
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -106.1658707, 273.3206482, -107.0443115, 275.3161011, -381.4819641, 380.3649597
1: -270.3559265, 414.1723328, -272.4507446, 417.1666565, -687.5223389, 686.6204224
2: -174.1107941, 404.1924438, -175.4979706, 407.0491028, -581.1598511, 579.6903687
3: -290.7397766, 479.1253967, -293.0501404, 482.5802002, -773.2598267, 771.9658813
4: -251.8461151, 463.3142090, -254.0505981, 466.6007385, -718.4468384, 717.3648071

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9666664, upper bound: 380.9670864
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9669742, upper bound: 380.9669742
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -105.5963135, 271.0330811, -110.3462219, 281.5076599, -387.1039734, 381.3793030
1: -266.8630371, 410.3604431, -280.0859375, 425.4723206, -692.3353271, 690.3627319
2: -172.2641449, 400.5419617, -181.0200806, 415.2046204, -587.4687500, 581.5620117
3: -288.9463196, 474.1787720, -301.7894287, 492.5526123, -781.1992188, 775.7467041
4: -251.0422058, 457.9292908, -262.3458252, 475.5311584, -726.5733032, 720.2750854

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9669934, upper bound: 380.9675764
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9665164, upper bound: 380.9672158
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -107.6284332, 277.2775574, -107.0443115, 275.3161011, -382.9445190, 384.3218689
1: -272.3641968, 420.2654114, -272.4507446, 417.1666565, -689.5308228, 692.7161865
2: -175.4547577, 409.7983093, -175.4979706, 407.0491028, -582.5038452, 585.2962036
3: -294.5341797, 485.4115906, -293.0501404, 482.5802002, -776.9350586, 778.4381104
4: -255.6973724, 469.0151062, -254.0505981, 466.6007385, -722.2980957, 723.0656738

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9666664, upper bound: 380.9671023
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9670752, upper bound: 380.9671051
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -98.5690384, 253.6150970, -109.0208817, 280.5858154, -379.1548462, 362.6359863
1: -248.4076385, 383.7231445, -275.8758240, 425.1848145, -673.5923462, 659.5989380
2: -160.1539612, 374.3469543, -177.7567749, 414.6723022, -574.8262329, 552.1037598
3: -269.6304932, 442.8076782, -298.2767639, 491.0883789, -760.7188721, 741.0844727
4: -234.3768005, 427.8878784, -259.1097412, 474.5095215, -708.8862915, 686.9976196

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9677124, upper bound: 380.9670743
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9677124, upper bound: 380.9678161
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -97.8052521, 251.8025513, -107.0402832, 275.4150085, -373.2202759, 358.8427429
1: -246.4061279, 381.0004272, -270.5291748, 417.3212585, -663.7273560, 651.5294800
2: -158.8500824, 371.6589966, -174.2568665, 407.0899658, -565.9400024, 545.9157715
3: -267.5203247, 439.6840820, -292.7090759, 482.0299683, -749.5502319, 732.3930054
4: -232.5729218, 424.8410950, -254.3705444, 465.6802368, -698.2531738, 679.2116699

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9653478, upper bound: 380.9652258
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9643438, upper bound: 380.9638453
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -107.9824600, 278.2383423, -109.0208817, 280.5858154, -388.5682678, 387.2592163
1: -273.4063110, 421.7153320, -275.8758240, 425.1848145, -698.5911255, 697.5911255
2: -176.1058960, 411.2939758, -177.7567749, 414.6723022, -590.7781982, 589.0507812
3: -295.5620422, 487.1115112, -298.2767639, 491.0883789, -786.6503296, 785.3883057
4: -256.5021362, 470.6853638, -259.1097412, 474.5095215, -731.0114746, 729.7951050

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9686845, upper bound: 380.9686824
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9686845, upper bound: 380.9686824
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -107.0666275, 276.0043640, -107.0402832, 275.4150085, -382.4816284, 383.0445557
1: -271.0520020, 418.3980408, -270.5291748, 417.3212585, -688.3731079, 688.9272461
2: -174.5509949, 407.9550476, -174.2568665, 407.0899658, -581.6409302, 582.2119141
3: -293.0571289, 483.2814941, -292.7090759, 482.0299683, -775.0870972, 775.9904175
4: -254.3557739, 466.9281006, -254.3705444, 465.6802368, -720.0360107, 721.2986450

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9667018, upper bound: 380.9665260
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9657294, upper bound: 380.9655786
time: 0.71 seconds

## Summary of splitting at layer (split count: 10)
- Time for NS candidates: 2.92 seconds
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 11, time: 2.92
Output dim: 0, lower bound: -380.9652307, upper bound: 380.9657366
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 11, time: 2.92
Output dim: 0, lower bound: -380.9652343, upper bound: 380.9651181
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.92
Output dim: 0, lower bound: -380.9666664, upper bound: 380.9670864
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.92
Output dim: 0, lower bound: -380.9669742, upper bound: 380.9669742
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 11, time: 2.92
Output dim: 0, lower bound: -380.9669934, upper bound: 380.9675764
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 2.92
Output dim: 0, lower bound: -380.9665164, upper bound: 380.9672158
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.92
Output dim: 0, lower bound: -380.9666664, upper bound: 380.9671023
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.92
Output dim: 0, lower bound: -380.9670752, upper bound: 380.9671051
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 11, time: 2.92
Output dim: 0, lower bound: -380.9677124, upper bound: 380.9670743
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 2.92
Output dim: 0, lower bound: -380.9677124, upper bound: 380.9678161
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 2.92
Output dim: 0, lower bound: -380.9653478, upper bound: 380.9652258
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 11, time: 2.92
Output dim: 0, lower bound: -380.9643438, upper bound: 380.9638453
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 11, time: 2.92
Output dim: 0, lower bound: -380.9686845, upper bound: 380.9686824
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 2.92
Output dim: 0, lower bound: -380.9686845, upper bound: 380.9686824
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.92
Output dim: 0, lower bound: -380.9667018, upper bound: 380.9665260
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.92
Output dim: 0, lower bound: -380.9657294, upper bound: 380.9655786

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -103.4392548, 265.3159790, -110.3462219, 281.5076599, -384.9468994, 375.6622009
1: -262.9247742, 401.7055359, -280.0859375, 425.4723206, -688.3804321, 681.5239868
2: -169.7039490, 392.3263855, -181.0200806, 415.2046204, -584.9084473, 573.3464355
3: -283.1882629, 464.9027405, -301.7894287, 492.5526123, -775.5444336, 766.1924438
4: -245.5809784, 449.2420349, -262.3458252, 475.5311584, -721.1120605, 711.5877686

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9645302, upper bound: 380.9648464
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9646240, upper bound: 380.9649295
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -105.1487579, 270.6260986, -106.8063202, 274.6888733, -379.8376465, 377.4324341
1: -267.5373535, 410.1239929, -271.7994080, 416.2241211, -683.7614136, 681.8858643
2: -172.3423920, 400.1627502, -175.0880280, 406.1229858, -578.4653931, 575.2507935
3: -287.9181824, 474.4165039, -292.3885803, 481.4831848, -769.3302002, 766.5692139
4: -249.5254211, 458.6178894, -253.5018005, 465.5168457, -715.0422363, 712.1196899

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9666664, upper bound: 380.9666664
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9666664, upper bound: 380.9669742
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -105.7390518, 272.2469482, -106.9766998, 275.1466980, -380.8857422, 379.2236328
1: -269.2177429, 412.5801697, -272.2698059, 416.9161377, -686.1337891, 684.8445435
2: -173.3596344, 402.5845642, -175.3803406, 406.7949829, -580.1546021, 577.9649048
3: -289.5597839, 477.2610779, -292.8634949, 482.2861938, -771.7902222, 769.9098511
4: -250.8433990, 461.4960327, -253.8922424, 466.3131714, -717.1563721, 715.3883057

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9669742, upper bound: 380.9666664
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9669742, upper bound: 380.9669742
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -104.3018494, 267.8535767, -109.8283310, 280.2434387, -384.5452881, 377.6819153
1: -263.6474915, 405.6939087, -278.8177490, 423.6233521, -687.2708130, 684.4088745
2: -170.1099396, 395.7878418, -180.1675415, 413.3133850, -583.4233398, 575.9553833
3: -285.4242859, 468.7431030, -300.3904724, 490.3944092, -775.5242310, 768.9019165
4: -247.9551544, 452.6639099, -261.1158142, 473.4443359, -721.3994751, 713.7797241

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663678, upper bound: 380.9670812
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663678, upper bound: 380.9672158
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -106.1113739, 272.2271729, -109.5061340, 279.3967285, -385.5081177, 381.7333069
1: -267.8779907, 412.2322998, -277.9606323, 422.2957153, -690.1734619, 690.0504761
2: -172.8270721, 402.0327454, -179.6326141, 412.0427856, -584.8697510, 581.6653442
3: -290.4395447, 476.2154236, -299.5423279, 488.8624268, -778.9039307, 775.5298462
4: -252.2658844, 459.8717041, -260.3561096, 471.9934082, -724.2592773, 720.2277832

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663678, upper bound: 380.9670812
time: 1.12 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663678, upper bound: 380.9672158
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -106.8974533, 275.3059387, -106.8063202, 274.6888733, -381.5863037, 382.1122437
1: -270.2631836, 417.2942200, -271.7994080, 416.2241211, -686.4870605, 689.0936279
2: -174.1740112, 406.8430176, -175.0880280, 406.1229858, -580.2969971, 581.9309692
3: -292.4697571, 481.9596863, -292.3885803, 481.4831848, -773.7600708, 774.2771606
4: -254.0475006, 465.5265808, -253.5018005, 465.5168457, -719.5643311, 719.0283813

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9666664, upper bound: 380.9667790
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9666664, upper bound: 380.9670964
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -107.2234421, 276.2514648, -106.9766998, 275.1466980, -382.3701477, 383.2281494
1: -271.2879944, 418.7528076, -272.2698059, 416.9161377, -688.2040405, 691.0225830
2: -174.7325592, 408.2673645, -175.3803406, 406.7949829, -581.5274658, 583.6477051
3: -293.4158936, 483.6332397, -292.8634949, 482.2861938, -775.5267334, 776.4685669
4: -254.7472992, 467.2826233, -253.8922424, 466.3131714, -721.0604248, 721.1748657

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9670752, upper bound: 380.9667934
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9670752, upper bound: 380.9671051
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -98.1344223, 252.5152588, -109.0208817, 280.5858154, -378.7202454, 361.5361023
1: -247.3265381, 382.0724792, -275.8758240, 425.1848145, -672.5113525, 657.9482422
2: -159.4436035, 372.7200317, -177.7567749, 414.6723022, -574.1159058, 550.4768066
3: -268.4645996, 440.8898315, -298.2767639, 491.0883789, -759.5529175, 739.1666260
4: -233.3524780, 426.0388184, -259.1097412, 474.5095215, -707.8619995, 685.1485596

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9652616, upper bound: 380.9642172
time: 1.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9660504, upper bound: 380.9654472
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -97.3319244, 250.1694641, -109.0208817, 280.5858154, -377.9177246, 359.1902466
1: -245.2926025, 378.3353271, -275.8758240, 425.1848145, -670.4774170, 654.2111206
2: -158.0515442, 369.2799683, -177.7567749, 414.6723022, -572.7238770, 547.0367432
3: -266.1616211, 436.7440491, -298.2767639, 491.0883789, -757.2500000, 735.0208130
4: -231.2778473, 422.0060730, -259.1097412, 474.5095215, -705.7872925, 681.1158447

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9652616, upper bound: 380.9651589
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9660504, upper bound: 380.9662890
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -107.6112442, 277.3066101, -109.0208817, 280.5858154, -388.1970520, 386.3274536
1: -272.4849854, 420.3139343, -275.8758240, 425.1848145, -697.6697998, 696.1896973
2: -175.5054321, 409.9263611, -177.7567749, 414.6723022, -590.1777344, 587.6830444
3: -294.5666809, 485.4862976, -298.2767639, 491.0883789, -785.6549683, 783.7630615
4: -255.6205444, 469.1278992, -259.1097412, 474.5095215, -730.1298828, 728.2376099

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9669200, upper bound: 380.9667057
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9669620, upper bound: 380.9669244
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -105.4610443, 271.7416382, -109.0208817, 280.5858154, -386.0468445, 380.7624817
1: -266.7455139, 411.8680115, -275.8758240, 425.1848145, -691.9302979, 687.7438354
2: -171.7047729, 401.7948608, -177.7567749, 414.6723022, -586.3770142, 579.5515747
3: -288.5463867, 475.7543030, -298.2767639, 491.0883789, -779.6347656, 774.0310669
4: -250.4638062, 459.6664124, -259.1097412, 474.5095215, -724.9731445, 718.7761230

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9669200, upper bound: 380.9667057
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9669620, upper bound: 380.9669244
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -106.2961349, 274.1233521, -106.7773285, 274.7783203, -381.0744629, 380.9006958
1: -269.0982666, 415.5945740, -269.8689575, 416.3697205, -685.4680176, 685.4635010
2: -173.2040863, 405.1884155, -173.8085022, 406.1543884, -579.3584595, 578.9968872
3: -290.9588318, 480.0336304, -291.9971313, 480.9309387, -771.8896484, 772.0307007
4: -252.4806213, 463.7976990, -253.7300110, 464.6228333, -717.1033325, 717.5277100

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9646090, upper bound: 380.9658209
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9667018, upper bound: 380.9665260
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -107.2181244, 277.0267334, -105.9028397, 272.6918335, -379.9099731, 382.9295654
1: -271.8059692, 420.1465149, -267.7960510, 413.2461548, -685.0521240, 687.9423828
2: -174.8230896, 409.6529846, -172.4136505, 403.1395874, -577.9624634, 582.0666504
3: -293.8058167, 485.3338928, -289.7468567, 477.3456421, -771.1514282, 774.8884277
4: -254.5607300, 469.0602112, -251.6031647, 461.1837158, -715.7444458, 720.6633301

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9638322, upper bound: 380.9648968
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9657085, upper bound: 380.9655786
time: 0.72 seconds

## Summary of splitting at layer (split count: 11)
- Time for NS candidates: 2.32 seconds
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9645302, upper bound: 380.9648464
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9646240, upper bound: 380.9649295
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9666664, upper bound: 380.9666664
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9666664, upper bound: 380.9669742
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9669742, upper bound: 380.9666664
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9669742, upper bound: 380.9669742
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9663678, upper bound: 380.9670812
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9663678, upper bound: 380.9672158
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9663678, upper bound: 380.9670812
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9663678, upper bound: 380.9672158
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9666664, upper bound: 380.9667790
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9666664, upper bound: 380.9670964
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9670752, upper bound: 380.9667934
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9670752, upper bound: 380.9671051
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9652616, upper bound: 380.9642172
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9660504, upper bound: 380.9654472
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9652616, upper bound: 380.9651589
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9660504, upper bound: 380.9662890
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9669200, upper bound: 380.9667057
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9669620, upper bound: 380.9669244
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9669200, upper bound: 380.9667057
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9669620, upper bound: 380.9669244
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9646090, upper bound: 380.9658209
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9667018, upper bound: 380.9665260
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9638322, upper bound: 380.9648968
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 12, time: 2.32
Output dim: 0, lower bound: -380.9657085, upper bound: 380.9655786

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -105.1487579, 270.6260986, -106.0083771, 272.5734253, -377.7221680, 376.6344604
1: -267.5373535, 410.1239929, -269.5859375, 413.0523071, -680.5896606, 679.6668091
2: -172.3423920, 400.1627502, -173.6976471, 402.9631653, -575.3055420, 573.8604126
3: -287.9181824, 474.4165039, -290.1737366, 477.8051758, -765.6433716, 764.3656616
4: -249.5254211, 458.6178894, -251.6785583, 461.8334351, -711.3588867, 710.2964478

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9639296, upper bound: 380.9650570
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9637160, upper bound: 380.9638983
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -105.1487579, 270.6260986, -106.6525497, 274.3337708, -379.4825439, 377.2786560
1: -267.5373535, 410.1239929, -271.4040833, 415.7135620, -683.2509155, 681.4788208
2: -172.3423920, 400.1627502, -174.8169861, 405.5765686, -577.9189453, 574.9797363
3: -287.9181824, 474.4165039, -291.9697266, 480.8760071, -768.6993408, 766.1510620
4: -249.5254211, 458.6178894, -253.1316986, 464.9352417, -714.4606934, 711.7495117

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9639296, upper bound: 380.9652241
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9637160, upper bound: 380.9640925
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -105.7390518, 272.2469482, -106.0083771, 272.5734253, -378.3124695, 378.2553101
1: -269.2177429, 412.5801697, -269.5859375, 413.0523071, -682.2699585, 682.1049805
2: -173.3596344, 402.5845642, -173.6976471, 402.9631653, -576.3228149, 576.2822266
3: -289.5597839, 477.2610779, -290.1737366, 477.8051758, -767.2761841, 767.1874390
4: -250.8433990, 461.4960327, -251.6785583, 461.8334351, -712.6767578, 713.1745605

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9641189, upper bound: 380.9648816
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9638803, upper bound: 380.9637160
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -105.7390518, 272.2469482, -106.6525497, 274.3337708, -380.0728149, 378.8995056
1: -269.2177429, 412.5801697, -271.4040833, 415.7135620, -684.9312744, 683.9777222
2: -173.3596344, 402.5845642, -174.8169861, 405.5765686, -578.9362183, 577.4015503
3: -289.5597839, 477.2610779, -291.9697266, 480.8760071, -770.3783569, 769.0180664
4: -250.8433990, 461.4960327, -253.1316986, 464.9352417, -715.7786255, 714.6277466

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9641189, upper bound: 380.9650055
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9638803, upper bound: 380.9637439
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -104.3018494, 267.8535767, -109.0369568, 278.3134460, -382.6152954, 376.8905334
1: -263.6474915, 405.6939087, -276.8810425, 420.8036194, -684.4510498, 682.4708862
2: -170.1099396, 395.7878418, -178.8651123, 410.4281921, -580.5381470, 574.6529541
3: -285.4242859, 468.7431030, -298.2514954, 487.1022949, -772.2211914, 766.7740479
4: -247.9551544, 452.6639099, -259.2340393, 470.2591248, -718.2142944, 711.8979492

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9660090, upper bound: 380.9659045
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9660090, upper bound: 380.9674503
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -104.3018494, 267.8535767, -110.8204041, 282.5547485, -386.8565674, 378.6739502
1: -263.6474915, 405.6939087, -281.0062866, 427.1182556, -690.7655640, 686.5893555
2: -170.1099396, 395.7878418, -181.6364288, 416.4691467, -586.5790405, 577.4242554
3: -285.4242859, 468.7431030, -303.1674805, 494.3470459, -779.4611206, 771.5553589
4: -247.9551544, 452.6639099, -263.5515137, 477.1993408, -725.1544800, 716.2154541

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9660090, upper bound: 380.9660040
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9660090, upper bound: 380.9675764
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -106.1113739, 272.2271729, -109.0369568, 278.3134460, -384.4248047, 381.2641296
1: -267.8779907, 412.2322998, -276.8810425, 420.8036194, -688.6814575, 688.9649048
2: -172.8270721, 402.0327454, -178.8651123, 410.4281921, -583.2552490, 580.8978271
3: -290.4395447, 476.2154236, -298.2514954, 487.1022949, -777.1185913, 774.2575684
4: -252.2658844, 459.8717041, -259.2340393, 470.2591248, -722.5250244, 719.1057129

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9661051, upper bound: 380.9663780
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9661051, upper bound: 380.9670812
time: 1.09 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -106.1113739, 272.2271729, -110.8204041, 282.5547485, -388.6660767, 383.0475769
1: -267.8779907, 412.2322998, -281.0062866, 427.1182556, -694.9960327, 693.0833740
2: -172.8270721, 402.0327454, -181.6364288, 416.4691467, -589.2961426, 583.6691895
3: -290.4395447, 476.2154236, -303.1674805, 494.3470459, -784.3585205, 779.0388794
4: -252.2658844, 459.8717041, -263.5515137, 477.1993408, -729.4652100, 723.4231567

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9661051, upper bound: 380.9665056
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9661051, upper bound: 380.9670812
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -106.8974533, 275.3059387, -106.0083771, 272.5734253, -379.4707947, 381.3143311
1: -270.2631836, 417.2942200, -269.5859375, 413.0523071, -683.3154297, 686.8801270
2: -174.1740112, 406.8430176, -173.6976471, 402.9631653, -577.1372070, 580.5406494
3: -292.4697571, 481.9596863, -290.1737366, 477.8051758, -770.0731812, 772.0736084
4: -254.0475006, 465.5265808, -251.6785583, 461.8334351, -715.8809204, 717.2051392

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9639394, upper bound: 380.9651128
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9637160, upper bound: 380.9640949
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -106.8974533, 275.3059387, -106.6525497, 274.3337708, -381.2312317, 381.9584961
1: -270.2631836, 417.2942200, -271.4040833, 415.7135620, -685.9767456, 688.6983032
2: -174.1740112, 406.8430176, -174.8169861, 405.5765686, -579.7506104, 581.6600342
3: -292.4697571, 481.9596863, -291.9697266, 480.8760071, -773.1292114, 773.8590698
4: -254.0475006, 465.5265808, -253.1316986, 464.9352417, -718.9827271, 718.6582642

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9639394, upper bound: 380.9652872
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9637160, upper bound: 380.9641645
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -107.2234421, 276.2514648, -106.0083771, 272.5734253, -379.7968445, 382.2598267
1: -271.2879944, 418.7528076, -269.5859375, 413.0523071, -684.3403320, 688.3387451
2: -174.7325592, 408.2673645, -173.6976471, 402.9631653, -577.6956787, 581.9650269
3: -293.4158936, 483.6332397, -290.1737366, 477.8051758, -771.0075073, 773.7485352
4: -254.7472992, 467.2826233, -251.6785583, 461.8334351, -716.5807495, 718.9611816

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9642408, upper bound: 380.9651239
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9638869, upper bound: 380.9637838
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -107.2234421, 276.2514648, -106.6525497, 274.3337708, -381.5571899, 382.9040222
1: -271.2879944, 418.7528076, -271.4040833, 415.7135620, -687.0015259, 690.1568604
2: -174.7325592, 408.2673645, -174.8169861, 405.5765686, -580.3090820, 583.0843506
3: -293.4158936, 483.6332397, -291.9697266, 480.8760071, -774.1148682, 775.5769043
4: -254.7472992, 467.2826233, -253.1316986, 464.9352417, -719.6825562, 720.4143066

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9642408, upper bound: 380.9652934
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9638869, upper bound: 380.9638871
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -98.0541992, 252.3157196, -108.5982742, 279.5174255, -377.5716248, 360.9139709
1: -247.1082153, 381.7780151, -274.7579346, 423.6067810, -670.7149658, 656.5359497
2: -159.3000793, 372.4230957, -177.0079956, 413.0719299, -572.3720093, 549.4309692
3: -268.2433167, 440.5429382, -297.1129150, 489.2319641, -757.4751587, 737.6558838
4: -233.1663666, 425.7009277, -258.1211243, 472.7028503, -705.8692017, 683.8219604

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9652451, upper bound: 380.9643625
time: 1.26 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9652451, upper bound: 380.9654472
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -97.2384186, 249.9367981, -108.5982742, 279.5174255, -376.7557983, 358.5350037
1: -245.0356293, 377.9945068, -274.7579346, 423.6067810, -668.6423950, 652.7523193
2: -157.8837128, 368.9335022, -177.0079956, 413.0719299, -570.9556274, 545.9414062
3: -265.9022827, 436.3392944, -297.1129150, 489.2319641, -755.1340332, 733.4522095
4: -231.0626831, 421.6085205, -258.1211243, 472.7028503, -703.7655029, 679.7295532

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9655397, upper bound: 380.9647701
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9655397, upper bound: 380.9662890
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -107.3846054, 276.7176819, -108.2682495, 278.5671692, -385.9517822, 384.9859314
1: -271.8716125, 419.4279785, -273.7191467, 422.1436462, -694.0151978, 693.1470947
2: -175.1224976, 409.0587769, -176.4310455, 411.6596375, -586.7821045, 585.4898071
3: -293.9388123, 484.4551392, -296.1475830, 487.5601501, -781.4989624, 780.6025391
4: -255.0943146, 468.1122131, -257.4068909, 470.9425354, -726.0368652, 725.5191040

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9677004, upper bound: 380.9676474
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9677004, upper bound: 380.9676476
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -107.5365372, 277.1178589, -108.5982742, 279.5174255, -387.0539551, 385.7160645
1: -272.2873840, 420.0355225, -274.7579346, 423.6067810, -695.8941650, 694.7934570
2: -175.3732605, 409.6439819, -177.0079956, 413.0719299, -588.4451904, 586.6519775
3: -294.3612366, 485.1590576, -297.1129150, 489.2319641, -783.5929565, 782.2719727
4: -255.4456787, 468.8092957, -258.1211243, 472.7028503, -728.1484985, 726.9304199

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9677004, upper bound: 380.9678291
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9677004, upper bound: 380.9678501
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -105.2359390, 271.1548767, -108.2682495, 278.5671692, -383.8030701, 379.4231262
1: -266.1386414, 410.9837036, -273.7191467, 422.1436462, -688.2822876, 684.7028809
2: -171.3250580, 400.9293823, -176.4310455, 411.6596375, -582.9846802, 577.3602905
3: -287.9196472, 474.7267151, -296.1475830, 487.5601501, -775.4797974, 770.8742676
4: -249.9372559, 458.6540527, -257.4068909, 470.9425354, -720.8797607, 716.0609131

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9668453, upper bound: 380.9667057
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9668453, upper bound: 380.9667057
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -105.3846741, 271.5502625, -108.5982742, 279.5174255, -384.9020996, 380.1484680
1: -266.5419922, 411.5888672, -274.7579346, 423.6067810, -690.1488037, 686.3467407
2: -171.5698853, 401.5102539, -177.0079956, 413.0719299, -584.6417847, 578.5181885
3: -288.3386536, 475.4243469, -297.1129150, 489.2319641, -777.5704346, 772.5372314
4: -250.2881165, 459.3434753, -258.1211243, 472.7028503, -722.9909668, 717.4645996

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9668482, upper bound: 380.9669244
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9668482, upper bound: 380.9669244
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -103.9322510, 268.2129822, -104.4026718, 269.0383606, -372.9706116, 372.6156616
1: -263.3911133, 406.6270447, -264.4379883, 407.8376160, -671.2287598, 671.0650635
2: -169.5277252, 396.6344910, -170.2536774, 398.0360718, -567.5637817, 566.8881836
3: -284.5928345, 469.6661987, -285.5633850, 471.1990662, -755.7918701, 755.2296143
4: -246.8096008, 454.0108337, -247.9169464, 455.5441589, -702.3537598, 701.9277954

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9637345, upper bound: 380.9632631
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9637345, upper bound: 380.9658209
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -106.0775375, 273.5794678, -106.2586823, 273.5115967, -379.5890808, 379.8381348
1: -268.5490723, 414.7725525, -268.5829773, 414.4601440, -683.0092163, 683.3555298
2: -172.8460999, 404.3854370, -172.9653015, 404.2971191, -577.1431885, 577.3507080
3: -290.3709106, 479.0807495, -290.6074829, 478.7186279, -769.0895386, 769.6651611
4: -251.9563446, 462.8874512, -252.4830933, 462.5199280, -714.4761963, 715.3704834

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9659189, upper bound: 380.9657042
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663674, upper bound: 380.9661635
time: 0.88 seconds

## Summary of splitting at layer (split count: 12)
- Time for NS candidates: 2.55 seconds
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9639296, upper bound: 380.9650570
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9637160, upper bound: 380.9638983
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9639296, upper bound: 380.9652241
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9637160, upper bound: 380.9640925
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9641189, upper bound: 380.9648816
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9638803, upper bound: 380.9637160
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9641189, upper bound: 380.9650055
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9638803, upper bound: 380.9637439
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9660090, upper bound: 380.9659045
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9660090, upper bound: 380.9674503
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9660090, upper bound: 380.9660040
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9660090, upper bound: 380.9675764
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9661051, upper bound: 380.9663780
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9661051, upper bound: 380.9670812
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9661051, upper bound: 380.9665056
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9661051, upper bound: 380.9670812
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9639394, upper bound: 380.9651128
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9637160, upper bound: 380.9640949
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9639394, upper bound: 380.9652872
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9637160, upper bound: 380.9641645
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9642408, upper bound: 380.9651239
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9638869, upper bound: 380.9637838
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9642408, upper bound: 380.9652934
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9638869, upper bound: 380.9638871
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9652451, upper bound: 380.9643625
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9652451, upper bound: 380.9654472
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9655397, upper bound: 380.9647701
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9655397, upper bound: 380.9662890
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9677004, upper bound: 380.9676474
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9677004, upper bound: 380.9676476
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9677004, upper bound: 380.9678291
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9677004, upper bound: 380.9678501
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9668453, upper bound: 380.9667057
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9668453, upper bound: 380.9667057
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9668482, upper bound: 380.9669244
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9668482, upper bound: 380.9669244
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9637345, upper bound: 380.9632631
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9637345, upper bound: 380.9658209
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9659189, upper bound: 380.9657042
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 13, time: 2.55
Output dim: 0, lower bound: -380.9663674, upper bound: 380.9661635

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -107.9018021, 275.7301331, -109.0369568, 278.3134460, -386.2152405, 384.7670898
1: -272.1459351, 417.2524414, -276.8810425, 420.8036194, -692.9494629, 694.1082764
2: -175.8283539, 406.7687683, -178.8651123, 410.4281921, -586.2565308, 585.6339111
3: -294.7842712, 482.2983093, -298.2514954, 487.1022949, -781.6236572, 780.3292847
4: -256.7862854, 465.0942688, -259.2340393, 470.2591248, -727.0453491, 724.3283081

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9644901, upper bound: 380.9642553
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9644367, upper bound: 380.9644139
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -105.5804138, 272.1296692, -109.0369568, 278.3134460, -383.8938599, 381.1303101
1: -267.1290283, 412.6765442, -276.8810425, 420.8036194, -687.8275146, 688.9144287
2: -172.0142059, 402.0872192, -178.8651123, 410.4281921, -582.4423828, 580.6812744
3: -288.9135742, 476.6084900, -298.2514954, 487.1022949, -775.6713867, 773.9736938
4: -250.8556061, 460.4017334, -259.2340393, 470.2591248, -721.1146240, 719.6357422

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9644901, upper bound: 380.9650264
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9644367, upper bound: 380.9651630
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -107.9018021, 275.7301331, -110.8204041, 282.5547485, -390.4565430, 386.5505371
1: -272.1459351, 417.2524414, -281.0062866, 427.1182556, -699.2640381, 698.2267456
2: -175.8283539, 406.7687683, -181.6364288, 416.4691467, -592.2973633, 588.4052124
3: -294.7842712, 482.2983093, -303.1674805, 494.3470459, -788.8635864, 785.1105347
4: -256.7862854, 465.0942688, -263.5515137, 477.1993408, -733.9855957, 728.6457520

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9646416, upper bound: 380.9644091
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9645283, upper bound: 380.9645015
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -105.5804138, 272.1296692, -110.8204041, 282.5547485, -388.1351624, 382.8686829
1: -267.1290283, 412.6765442, -281.0062866, 427.1182556, -694.0890503, 693.0329590
2: -172.0142059, 402.0872192, -181.6364288, 416.4691467, -588.4832764, 583.4444580
3: -288.9135742, 476.6084900, -303.1674805, 494.3470459, -782.9112549, 778.7549438
4: -250.8556061, 460.4017334, -263.5515137, 477.1993408, -728.0548096, 723.9531250

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9646416, upper bound: 380.9651779
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9645283, upper bound: 380.9652647
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -109.2073288, 278.8616638, -109.0369568, 278.3134460, -387.5207825, 387.8986206
1: -275.1231995, 421.9192200, -276.8810425, 420.8036194, -695.9267578, 698.7271729
2: -177.8171539, 411.2214355, -178.8651123, 410.4281921, -588.2453003, 590.0865479
3: -298.4091492, 487.6341248, -298.2514954, 487.1022949, -785.1118774, 785.6712036
4: -259.9255981, 470.2721252, -259.2340393, 470.2591248, -730.1846924, 729.5061646

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9646004, upper bound: 380.9645584
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9645512, upper bound: 380.9647322
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -107.4883041, 276.8266602, -109.0369568, 278.3134460, -385.8017578, 385.8587646
1: -271.7004395, 419.7166443, -276.8810425, 420.8036194, -692.3601685, 695.8954468
2: -174.9011078, 408.8186951, -178.8651123, 410.4281921, -585.3292847, 587.4660645
3: -294.2221985, 484.6556091, -298.2514954, 487.1022949, -780.8661499, 782.0126343
4: -255.3754425, 468.1961670, -259.2340393, 470.2591248, -725.6345825, 727.4301758

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9646004, upper bound: 380.9649064
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9645512, upper bound: 380.9650638
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -109.2073288, 278.8616638, -110.8204041, 282.5547485, -391.7620850, 389.6820374
1: -275.1231995, 421.9192200, -281.0062866, 427.1182556, -702.2413330, 702.8456421
2: -177.8171539, 411.2214355, -181.6364288, 416.4691467, -594.2861938, 592.8578491
3: -298.4091492, 487.6341248, -303.1674805, 494.3470459, -792.3518066, 790.4524536
4: -259.9255981, 470.2721252, -263.5515137, 477.1993408, -737.1249390, 733.8236084

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9647587, upper bound: 380.9647254
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9646467, upper bound: 380.9648101
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -107.4883041, 276.8266602, -110.8204041, 282.5547485, -390.0430603, 387.5971375
1: -271.7004395, 419.7166443, -281.0062866, 427.1182556, -698.6217041, 700.0138550
2: -174.9011078, 408.8186951, -181.6364288, 416.4691467, -591.3702393, 590.2292480
3: -294.2221985, 484.6556091, -303.1674805, 494.3470459, -788.1060791, 786.7938843
4: -255.3754425, 468.1961670, -263.5515137, 477.1993408, -732.5747681, 731.7476807

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9647587, upper bound: 380.9649842
time: 1.17 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9646467, upper bound: 380.9650638
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -96.8022537, 248.8534698, -108.5982742, 279.5174255, -376.3196716, 357.4516296
1: -243.8311768, 376.4077148, -274.7579346, 423.6067810, -667.4379883, 651.1655884
2: -157.0986023, 367.3196716, -177.0079956, 413.0719299, -570.1705322, 544.3275757
3: -264.6895752, 434.4588928, -297.1129150, 489.2319641, -753.9215088, 731.5717773
4: -230.0586853, 419.7621460, -258.1211243, 472.7028503, -702.7614136, 677.8831787

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -106.8158569, 275.1766052, -108.2682495, 278.5671692, -385.3830261, 383.4448547
1: -270.2198792, 417.1106873, -273.7191467, 422.1436462, -692.3635254, 690.8298340
2: -174.1098328, 406.7485352, -176.4310455, 411.6596375, -585.7694702, 583.1795654
3: -292.3235779, 481.7680969, -296.1475830, 487.5601501, -779.8837280, 777.9133301
4: -253.8187714, 465.3789673, -257.4068909, 470.9425354, -724.7612915, 722.7858887

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9658785, upper bound: 380.9647612
time: 1.13 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9675455, upper bound: 380.9674868
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9674368, upper bound: 380.9674313
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -107.1718979, 276.1944580, -108.2682495, 278.5671692, -385.7390137, 384.4627075
1: -271.3210754, 418.6730042, -273.7191467, 422.1436462, -693.4645996, 692.3921509
2: -174.7265015, 408.2633057, -176.4310455, 411.6596375, -586.3860474, 584.6942139
3: -293.3555603, 483.5572815, -296.1475830, 487.5601501, -780.9157104, 779.7046509
4: -254.5909729, 467.2487183, -257.4068909, 470.9425354, -725.5335083, 724.6556396

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9658785, upper bound: 380.9647871
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9675455, upper bound: 380.9675084
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9674368, upper bound: 380.9674723
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -106.8158569, 275.1766052, -108.5982742, 279.5174255, -386.3332825, 383.7748108
1: -270.2198792, 417.1106873, -274.7579346, 423.6067810, -693.8266602, 691.8686523
2: -174.1098328, 406.7485352, -177.0079956, 413.0719299, -587.1817627, 583.7565308
3: -292.3235779, 481.7680969, -297.1129150, 489.2319641, -781.5555420, 778.8779297
4: -253.8187714, 465.3789673, -258.1211243, 472.7028503, -726.5215454, 723.5000000

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9658363, upper bound: 380.9648929
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9675068, upper bound: 380.9676722
time: 1.47 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -107.1718979, 276.1944580, -108.5982742, 279.5174255, -386.6892700, 384.7926636
1: -271.3210754, 418.6730042, -274.7579346, 423.6067810, -694.9277954, 693.4309082
2: -174.7265015, 408.2633057, -177.0079956, 413.0719299, -587.7983398, 585.2712402
3: -293.3555603, 483.5572815, -297.1129150, 489.2319641, -782.5875244, 780.6701660
4: -254.5909729, 467.2487183, -258.1211243, 472.7028503, -727.2937622, 725.3697510

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9658363, upper bound: 380.9649267
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9675068, upper bound: 380.9676464
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -104.5499268, 269.3464661, -108.2682495, 278.5671692, -383.1170959, 377.6147156
1: -264.2265015, 408.2659607, -273.7191467, 422.1436462, -686.3701172, 681.9851074
2: -170.1376953, 398.2440491, -176.4310455, 411.6596375, -581.7972412, 574.6751099
3: -285.9730835, 471.5877075, -296.1475830, 487.5601501, -773.5332031, 767.7350464
4: -248.3546448, 455.5066528, -257.4068909, 470.9425354, -719.2971802, 712.9135742

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9652984, upper bound: 380.9644852
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9664831, upper bound: 380.9664737
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663263, upper bound: 380.9663263
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -105.0193863, 270.6366882, -108.2682495, 278.5671692, -383.5865479, 378.9049072
1: -265.5698242, 410.2573242, -273.7191467, 422.1436462, -687.7135010, 683.9764404
2: -170.9252014, 400.1545410, -176.4310455, 411.6596375, -582.5847778, 576.5853882
3: -287.3452454, 473.8517151, -296.1475830, 487.5601501, -774.9053955, 769.9991455
4: -249.4468994, 457.8037415, -257.4068909, 470.9425354, -720.3894043, 715.2106323

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9652984, upper bound: 380.9644852
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9664831, upper bound: 380.9664737
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663263, upper bound: 380.9663263
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -104.5499268, 269.3464661, -108.5982742, 279.5174255, -384.0673523, 377.9447327
1: -264.2265015, 408.2659607, -274.7579346, 423.6067810, -687.8332520, 683.0239258
2: -170.1376953, 398.2440491, -177.0079956, 413.0719299, -583.2095337, 575.2520752
3: -285.9730835, 471.5877075, -297.1129150, 489.2319641, -775.2049561, 768.7005615
4: -248.3546448, 455.5066528, -258.1211243, 472.7028503, -721.0574341, 713.6277466

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9651435, upper bound: 380.9645020
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9664721, upper bound: 380.9666830
time: 1.34 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -105.0193863, 270.6366882, -108.5982742, 279.5174255, -384.5368042, 379.2348633
1: -265.5698242, 410.2573242, -274.7579346, 423.6067810, -689.1766357, 685.0151978
2: -170.9252014, 400.1545410, -177.0079956, 413.0719299, -583.9970703, 577.1624756
3: -287.3452454, 473.8517151, -297.1129150, 489.2319641, -776.5771484, 770.9645996
4: -249.4468994, 457.8037415, -258.1211243, 472.7028503, -722.1497192, 715.9248047

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9651435, upper bound: 380.9644631
time: 1.17 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9664721, upper bound: 380.9664996
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -105.5668411, 272.3659668, -104.4026718, 269.0383606, -374.6051636, 376.7686462
1: -267.3048096, 412.9611511, -264.4379883, 407.8376160, -675.1424561, 677.3989868
2: -172.0235596, 402.6370239, -170.2536774, 398.0360718, -570.0596313, 572.8906860
3: -288.9967041, 476.9873962, -285.5633850, 471.1990662, -760.1098633, 762.4788208
4: -250.7167816, 460.9111633, -247.9169464, 455.5441589, -706.2609253, 708.8280640

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9633864, upper bound: 380.9655975
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9633864, upper bound: 380.9658209
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -106.8465805, 273.1825562, -103.4137650, 265.1227417, -371.9692383, 376.5963135
1: -269.4468079, 413.3783569, -260.9583435, 401.3289185, -670.7755737, 674.3366089
2: -174.0614624, 403.0294800, -168.3786163, 391.7956543, -565.8571167, 571.4080811
3: -291.9259338, 477.8542175, -282.7090759, 463.7876282, -755.7135620, 760.2468872
4: -254.2854004, 460.7806702, -245.8765106, 447.8184814, -702.1038818, 706.6571655

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9659189, upper bound: 380.9657042
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9659189, upper bound: 380.9657042
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -104.5216751, 269.5527649, -105.4973526, 271.5827637, -376.1044312, 375.0500793
1: -264.4342346, 408.7763367, -266.5801086, 411.6060791, -676.0401611, 675.3564453
2: -170.2042542, 398.3418884, -171.7053680, 401.4067383, -571.6109009, 570.0472412
3: -286.0435181, 472.1462708, -288.5288391, 475.4100647, -761.4534302, 760.5360107
4: -248.3271790, 456.0674133, -250.7320404, 459.2636719, -707.5908203, 706.7993164

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663674, upper bound: 380.9661635
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663674, upper bound: 380.9661635
time: 0.84 seconds

## Summary of splitting at layer (split count: 13)
- Time for NS candidates: 2.54 seconds
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9644901, upper bound: 380.9642553
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9644367, upper bound: 380.9644139
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9644901, upper bound: 380.9650264
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9644367, upper bound: 380.9651630
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9646416, upper bound: 380.9644091
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9645283, upper bound: 380.9645015
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9646416, upper bound: 380.9651779
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9645283, upper bound: 380.9652647
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9646004, upper bound: 380.9645584
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9645512, upper bound: 380.9647322
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9646004, upper bound: 380.9649064
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9645512, upper bound: 380.9650638
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9647587, upper bound: 380.9647254
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9646467, upper bound: 380.9648101
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9647587, upper bound: 380.9649842
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9646467, upper bound: 380.9650638
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9675455, upper bound: 380.9674868
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9674368, upper bound: 380.9674313
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9675455, upper bound: 380.9675084
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9674368, upper bound: 380.9674723
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9664831, upper bound: 380.9664737
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9663263, upper bound: 380.9663263
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9664831, upper bound: 380.9664737
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9663263, upper bound: 380.9663263
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9633864, upper bound: 380.9655975
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9633864, upper bound: 380.9658209
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9659189, upper bound: 380.9657042
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9659189, upper bound: 380.9657042
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9663674, upper bound: 380.9661635
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 14, time: 2.54
Output dim: 0, lower bound: -380.9663674, upper bound: 380.9661635

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -106.8158569, 275.1766052, -108.1808472, 278.3468933, -385.1627502, 383.3574524
1: -270.2198792, 417.1106873, -273.4933167, 421.8181152, -692.0379639, 690.6040039
2: -174.1098328, 406.7485352, -176.2826385, 411.3313293, -585.4411621, 583.0311890
3: -292.3235779, 481.7680969, -295.9106750, 487.1809998, -779.5045776, 777.6771240
4: -253.8187714, 465.3789673, -257.1983948, 470.5711975, -724.3899536, 722.5772095

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9674368, upper bound: 380.9674313
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9674368, upper bound: 380.9674313
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -106.8158569, 275.1766052, -108.4967728, 279.1310120, -385.9468384, 383.6733398
1: -270.2198792, 417.1106873, -274.3739319, 422.9880066, -693.2078857, 691.4846191
2: -174.1098328, 406.7485352, -176.8286896, 412.4943237, -586.6041260, 583.5772095
3: -292.3235779, 481.7680969, -296.8176575, 488.5641479, -780.8876953, 778.5786743
4: -253.8187714, 465.3789673, -257.9172668, 471.9305725, -725.7493286, 723.2962646

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9674368, upper bound: 380.9674313
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9674368, upper bound: 380.9674313
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -107.1718979, 276.1944580, -108.1808472, 278.3468933, -385.5187378, 384.3753052
1: -271.3210754, 418.6730042, -273.4933167, 421.8181152, -693.1389771, 692.1662598
2: -174.7265015, 408.2633057, -176.2826385, 411.3313293, -586.0576782, 584.5458374
3: -293.3555603, 483.5572815, -295.9106750, 487.1809998, -780.5365601, 779.4679565
4: -254.5909729, 467.2487183, -257.1983948, 470.5711975, -725.1621704, 724.4470215

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9675946, upper bound: 380.9674366
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9675946, upper bound: 380.9674723
time: 1.13 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -107.1718979, 276.1944580, -108.4967728, 279.1310120, -386.3028259, 384.6911926
1: -271.3210754, 418.6730042, -274.3739319, 422.9880066, -694.3090210, 693.0469360
2: -174.7265015, 408.2633057, -176.8286896, 412.4943237, -587.2208252, 585.0919800
3: -293.3555603, 483.5572815, -296.8176575, 488.5641479, -781.9196777, 780.3715820
4: -254.5909729, 467.2487183, -257.9172668, 471.9305725, -726.5214844, 725.1660156

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9675946, upper bound: 380.9674366
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9675946, upper bound: 380.9674723
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -104.5499268, 269.3464661, -108.1808472, 278.3468933, -382.8968201, 377.5273132
1: -264.2265015, 408.2659607, -273.4933167, 421.8181152, -686.0444946, 681.7592773
2: -170.1376953, 398.2440491, -176.2826385, 411.3313293, -581.4688721, 574.5266724
3: -285.9730835, 471.5877075, -295.9106750, 487.1809998, -773.1540527, 767.4982910
4: -248.3546448, 455.5066528, -257.1983948, 470.5711975, -718.9258423, 712.7049561

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663383, upper bound: 380.9664232
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663383, upper bound: 380.9664232
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -104.5499268, 269.3464661, -108.4967728, 279.1310120, -383.6809387, 377.8432312
1: -264.2265015, 408.2659607, -274.3739319, 422.9880066, -687.2144775, 682.6398926
2: -170.1376953, 398.2440491, -176.8286896, 412.4943237, -582.6320190, 575.0727539
3: -285.9730835, 471.5877075, -296.8176575, 488.5641479, -774.5372314, 768.4052734
4: -248.3546448, 455.5066528, -257.9172668, 471.9305725, -720.2851562, 713.4239502

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663383, upper bound: 380.9664232
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663383, upper bound: 380.9664232
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -105.0193863, 270.6366882, -108.1808472, 278.3468933, -383.3662720, 378.8175354
1: -265.5698242, 410.2573242, -273.4933167, 421.8181152, -687.3877563, 683.7506104
2: -170.9252014, 400.1545410, -176.2826385, 411.3313293, -582.2564087, 576.4370728
3: -287.3452454, 473.8517151, -295.9106750, 487.1809998, -774.5261841, 769.7623901
4: -249.4468994, 457.8037415, -257.1983948, 470.5711975, -720.0180664, 715.0020752

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663263, upper bound: 380.9663263
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663263, upper bound: 380.9663263
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -105.0193863, 270.6366882, -108.4967728, 279.1310120, -384.1503906, 379.1334229
1: -265.5698242, 410.2573242, -274.3739319, 422.9880066, -688.5578613, 684.6312256
2: -170.9252014, 400.1545410, -176.8286896, 412.4943237, -583.4194946, 576.9831543
3: -287.3452454, 473.8517151, -296.8176575, 488.5641479, -775.9093628, 770.6693115
4: -249.4468994, 457.8037415, -257.9172668, 471.9305725, -721.3774414, 715.7210083

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663263, upper bound: 380.9663263
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9663263, upper bound: 380.9663263
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -105.5668411, 272.3659668, -105.1590271, 271.5427856, -377.1095581, 377.5249939
1: -267.3048096, 412.9611511, -266.6814270, 411.7982483, -679.1030273, 679.6423340
2: -172.0235596, 402.6370239, -171.4926605, 401.9024048, -573.9259644, 574.1294556
3: -288.9967041, 476.9873962, -287.9636230, 475.8216248, -764.4935303, 764.8291016
4: -250.7167816, 460.9111633, -249.5786133, 460.1263733, -710.8431396, 710.4896851

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9623692, upper bound: 380.9644169
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9627477, upper bound: 380.9642072
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -106.8465805, 273.1825562, -102.9187927, 263.9133911, -370.7599487, 376.1013489
1: -269.4468079, 413.3783569, -259.7203674, 399.5131531, -668.9598999, 673.0987549
2: -174.0614624, 403.0294800, -167.5483856, 390.0145264, -564.0759888, 570.5778198
3: -291.9259338, 477.8542175, -281.3677368, 461.6856079, -753.6115112, 758.9030151
4: -254.2854004, 460.7806702, -244.6786041, 445.8054810, -700.0906982, 705.4592896

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9620042, upper bound: 380.9626390
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9656181, upper bound: 380.9651911
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9654659, upper bound: 380.9650349
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -106.8465805, 273.1825562, -103.8688965, 266.9082947, -373.7547913, 377.0514526
1: -269.4468079, 413.3783569, -262.5077209, 404.2235413, -673.6702881, 675.8860474
2: -174.0614624, 403.0294800, -169.1595154, 394.6371765, -568.6986084, 572.1889648
3: -291.9259338, 477.8542175, -284.2920227, 467.1596069, -758.8682861, 761.7994385
4: -254.2854004, 460.7806702, -246.7905884, 451.2367554, -705.5221558, 707.5712280

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9620042, upper bound: 380.9626390
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9656181, upper bound: 380.9651911
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9654659, upper bound: 380.9650349
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -104.5216751, 269.5527649, -105.0225220, 270.4331665, -374.9548340, 374.5752258
1: -264.4342346, 408.7763367, -265.3945923, 409.8877563, -674.3218384, 674.1708374
2: -170.2042542, 398.3418884, -170.9040375, 399.7175903, -569.9218750, 569.2459106
3: -286.0435181, 472.1462708, -287.2456970, 473.4267883, -759.4703369, 759.2496948
4: -248.3271790, 456.0674133, -249.5778656, 457.3563843, -705.6835938, 705.6452026

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9626171, upper bound: 380.9631164
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9624890, upper bound: 380.9619504
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -104.5216751, 269.5527649, -106.0982056, 273.7059326, -378.2276001, 375.6509399
1: -264.4342346, 408.7763367, -268.4276733, 414.9992065, -679.4333496, 677.2038574
2: -170.2042542, 398.3418884, -172.6994324, 404.7118530, -574.9161377, 571.0412598
3: -286.0435181, 472.1462708, -290.5155945, 479.3538513, -765.3972778, 762.4700317
4: -248.3271790, 456.0674133, -252.0200195, 463.2041626, -711.5313110, 708.0873413

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9626171, upper bound: 380.9631164
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9624890, upper bound: 380.9619504
time: 0.72 seconds

## Summary of splitting at layer (split count: 14)
- Time for NS candidates: 3.35 seconds
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9674368, upper bound: 380.9674313
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9674368, upper bound: 380.9674313
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9674368, upper bound: 380.9674313
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9674368, upper bound: 380.9674313
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9675946, upper bound: 380.9674366
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9675946, upper bound: 380.9674723
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9675946, upper bound: 380.9674366
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9675946, upper bound: 380.9674723
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9663383, upper bound: 380.9664232
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9663383, upper bound: 380.9664232
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9663383, upper bound: 380.9664232
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9663383, upper bound: 380.9664232
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9663263, upper bound: 380.9663263
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9663263, upper bound: 380.9663263
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9663263, upper bound: 380.9663263
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9663263, upper bound: 380.9663263
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9623692, upper bound: 380.9644169
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9627477, upper bound: 380.9642072
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9656181, upper bound: 380.9651911
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9654659, upper bound: 380.9650349
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9656181, upper bound: 380.9651911
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9654659, upper bound: 380.9650349
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9626171, upper bound: 380.9631164
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9624890, upper bound: 380.9619504
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9626171, upper bound: 380.9631164
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 15, time: 3.35
Output dim: 0, lower bound: -380.9624890, upper bound: 380.9619504

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -106.7286377, 274.9567261, -108.1808472, 278.3468933, -385.0755310, 383.1375732
1: -269.9944458, 416.7856750, -273.4933167, 421.8181152, -691.8125000, 690.2789307
2: -173.9616699, 406.4205933, -176.2826385, 411.3313293, -585.2929688, 582.7031250
3: -292.0871887, 481.3895264, -295.9106750, 487.1809998, -779.2681274, 777.2991333
4: -253.6107941, 465.0086365, -257.1983948, 470.5711975, -724.1819458, 722.2070312

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -107.0451660, 275.7438660, -108.1808472, 278.3468933, -385.3920593, 383.9247131
1: -270.8770142, 417.9592590, -273.4933167, 421.8181152, -692.6949463, 691.4525146
2: -174.5094910, 407.5880737, -176.2826385, 411.3313293, -585.8407593, 583.8706665
3: -292.9952698, 482.7769165, -295.9106750, 487.1809998, -780.1761475, 778.6876221
4: -254.3326416, 466.3730164, -257.1983948, 470.5711975, -724.9038086, 723.5712891

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -106.7286377, 274.9567261, -108.4967728, 279.1310120, -385.8596497, 383.4534912
1: -269.9944458, 416.7856750, -274.3739319, 422.9880066, -692.9824219, 691.1596069
2: -173.9616699, 406.4205933, -176.8286896, 412.4943237, -586.4559937, 583.2492065
3: -292.0871887, 481.3895264, -296.8176575, 488.5641479, -780.6513062, 778.2007446
4: -253.6107941, 465.0086365, -257.9172668, 471.9305725, -725.5411987, 722.9259033

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -107.0451660, 275.7438660, -108.4967728, 279.1310120, -386.1761780, 384.2406311
1: -270.8770142, 417.9592590, -274.3739319, 422.9880066, -693.8649902, 692.3331909
2: -174.5094910, 407.5880737, -176.8286896, 412.4943237, -587.0037842, 584.4167480
3: -292.9952698, 482.7769165, -296.8176575, 488.5641479, -781.5593262, 779.5946045
4: -254.3326416, 466.3730164, -257.9172668, 471.9305725, -726.2631836, 724.2902832

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -107.0843430, 275.9722900, -108.1808472, 278.3468933, -385.4311829, 384.1531372
1: -271.0935059, 418.3439636, -273.4933167, 421.8181152, -692.9114990, 691.8372192
2: -174.5767822, 407.9311218, -176.2826385, 411.3313293, -585.9080811, 584.2136841
3: -293.1176758, 483.1737976, -295.9106750, 487.1809998, -780.2986450, 779.0844727
4: -254.3829956, 466.8728333, -257.1983948, 470.5711975, -724.9541626, 724.0711670

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -107.3340836, 276.5950012, -108.1808472, 278.3468933, -385.6809692, 384.7758484
1: -271.7980042, 419.2736816, -273.4933167, 421.8181152, -693.6160278, 692.7669678
2: -175.0099640, 408.8595886, -176.2826385, 411.3313293, -586.3413086, 585.1422119
3: -293.8353271, 484.2785339, -295.9106750, 487.1809998, -781.0163574, 780.1892090
4: -254.9458160, 467.9621887, -257.1983948, 470.5711975, -725.5170288, 725.1604614

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -107.0843430, 275.9722900, -108.4967728, 279.1310120, -386.2152710, 384.4690552
1: -271.0935059, 418.3439636, -274.3739319, 422.9880066, -694.0815430, 692.7178955
2: -174.5767822, 407.9311218, -176.8286896, 412.4943237, -587.0711060, 584.7597656
3: -293.1176758, 483.1737976, -296.8176575, 488.5641479, -781.6818237, 779.9890137
4: -254.3829956, 466.8728333, -257.9172668, 471.9305725, -726.3134766, 724.7901001

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -107.3340836, 276.5950012, -108.4967728, 279.1310120, -386.4650574, 385.0917664
1: -271.7980042, 419.2736816, -274.3739319, 422.9880066, -694.7860107, 693.6475830
2: -175.0099640, 408.8595886, -176.8286896, 412.4943237, -587.5042725, 585.6882935
3: -293.8353271, 484.2785339, -296.8176575, 488.5641479, -782.3994751, 781.0961914
4: -254.9458160, 467.9621887, -257.9172668, 471.9305725, -726.8764038, 725.8794556

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -104.4645462, 269.1311035, -108.1808472, 278.3468933, -382.8114319, 377.3119507
1: -264.0054016, 407.9475708, -273.4933167, 421.8181152, -685.8233643, 681.4407959
2: -169.9925232, 397.9225769, -176.2826385, 411.3313293, -581.3237915, 574.2052002
3: -285.7415466, 471.2159424, -295.9106750, 487.1809998, -772.9224854, 767.1265869
4: -248.1513672, 455.1436157, -257.1983948, 470.5711975, -718.7225342, 712.3419189

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -104.7783508, 269.9157715, -108.1808472, 278.3468933, -383.1252441, 378.0966187
1: -264.8817444, 409.1163940, -273.4933167, 421.8181152, -686.6998291, 682.6096802
2: -170.5356140, 399.0907593, -176.2826385, 411.3313293, -581.8668823, 575.3732300
3: -286.6442261, 472.6018982, -295.9106750, 487.1809998, -773.8250732, 768.5125732
4: -248.8666992, 456.5075684, -257.1983948, 470.5711975, -719.4378662, 713.7059326

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -104.4645462, 269.1311035, -108.4967728, 279.1310120, -383.5955200, 377.6278381
1: -264.0054016, 407.9475708, -274.3739319, 422.9880066, -686.9934082, 682.3215332
2: -169.9925232, 397.9225769, -176.8286896, 412.4943237, -582.4868164, 574.7512817
3: -285.7415466, 471.2159424, -296.8176575, 488.5641479, -774.3056641, 768.0335693
4: -248.1513672, 455.1436157, -257.9172668, 471.9305725, -720.0817871, 713.0609131

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -104.7783508, 269.9157715, -108.4967728, 279.1310120, -383.9093628, 378.4125366
1: -264.8817444, 409.1163940, -274.3739319, 422.9880066, -687.8697510, 683.4903564
2: -170.5356140, 399.0907593, -176.8286896, 412.4943237, -583.0299072, 575.9193115
3: -286.6442261, 472.6018982, -296.8176575, 488.5641479, -775.2082520, 769.4195557
4: -248.8666992, 456.5075684, -257.9172668, 471.9305725, -720.7972412, 714.4248047

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -104.9384842, 270.4325256, -108.1808472, 278.3468933, -383.2853699, 378.6133728
1: -265.3600769, 409.9557495, -273.4933167, 421.8181152, -687.1781006, 683.4490356
2: -170.7872620, 399.8498535, -176.2826385, 411.3313293, -582.1185913, 576.1323853
3: -287.1259155, 473.4993286, -295.9106750, 487.1809998, -774.3068848, 769.4100342
4: -249.2545776, 457.4590759, -257.1983948, 470.5711975, -719.8258057, 714.6574097

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -105.1514893, 270.9686890, -108.1808472, 278.3468933, -383.4983215, 379.1495361
1: -265.9672852, 410.7560730, -273.4933167, 421.8181152, -687.7852173, 684.2492676
2: -171.1592560, 400.6481934, -176.2826385, 411.3313293, -582.4905396, 576.9307861
3: -287.7471924, 474.4543457, -295.9106750, 487.1809998, -774.9281616, 770.3649902
4: -249.7380676, 458.3990479, -257.1983948, 470.5711975, -720.3092651, 715.5974121

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -104.9384842, 270.4325256, -108.4967728, 279.1310120, -384.0694580, 378.9292603
1: -265.3600769, 409.9557495, -274.3739319, 422.9880066, -688.3480835, 684.3297119
2: -170.7872620, 399.8498535, -176.8286896, 412.4943237, -583.2816162, 576.6784668
3: -287.1259155, 473.4993286, -296.8176575, 488.5641479, -775.6900635, 770.3170166
4: -249.2545776, 457.4590759, -257.9172668, 471.9305725, -721.1851807, 715.3763428

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -105.1514893, 270.9686890, -108.4967728, 279.1310120, -384.2824097, 379.4654541
1: -265.9672852, 410.7560730, -274.3739319, 422.9880066, -688.9552612, 685.1300049
2: -171.1592560, 400.6481934, -176.8286896, 412.4943237, -583.6535645, 577.4768677
3: -287.7471924, 474.4543457, -296.8176575, 488.5641479, -776.3113403, 771.2719727
4: -249.7380676, 458.3990479, -257.9172668, 471.9305725, -721.6685181, 716.3162842

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.78 + 322.79 = 325.56 seconds
