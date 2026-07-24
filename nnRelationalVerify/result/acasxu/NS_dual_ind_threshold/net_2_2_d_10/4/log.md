## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 339.41632513516


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-149.8139038, 226.3787994, -149.8139038, 226.3787994, -376.1926880, 376.1926880)
1: (-118.9682159, 218.5465088, -118.9682159, 218.5465088, -337.5147095, 337.5147095)
2: (-102.7016296, 224.2848206, -102.7016296, 224.2848206, -326.9864502, 326.9864502)
3: (-155.0733032, 221.6566010, -155.0733032, 221.6566010, -376.7298584, 376.7298584)
4: (-123.6448669, 237.1756439, -123.6448669, 237.1756439, -360.8204956, 360.8204956)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.57 + 2.09 = 3.66 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -339.4332968, upper bound: 339.4332968

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4307018, upper bound: 339.4289571
time: 0.89 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4310175, upper bound: 339.4310175
time: 0.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.96 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.96
Output dim: 4, lower bound: -339.4307018, upper bound: 339.4289571
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.96
Output dim: 4, lower bound: -339.4310175, upper bound: 339.4310175

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -119.5369644, 181.6595764, -149.5541382, 225.9937897, -345.5307312, 331.2137146
1: -95.3491898, 176.5832367, -118.7654724, 218.1844788, -313.5335999, 295.3486938
2: -82.4457779, 180.8220520, -102.5277634, 223.9108887, -306.3566284, 283.3498230
3: -124.2547455, 178.3655853, -154.8089142, 221.2846527, -345.5393677, 333.1744995
4: -99.1074524, 191.3053741, -123.4344711, 236.7805786, -335.8880005, 314.7398071

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4289571, upper bound: 339.4289571
time: 0.91 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4289571, upper bound: 339.4289571
time: 0.85 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -142.2355347, 215.0777283, -149.8139038, 226.3787994, -368.6143188, 364.8916321
1: -113.0008011, 207.8044281, -118.9682159, 218.5465088, -331.5473022, 326.7725525
2: -97.5341110, 213.1814423, -102.7016296, 224.2848206, -321.8188782, 315.8830261
3: -147.2916718, 210.6535492, -155.0733032, 221.6566010, -368.9482422, 365.7268677
4: -117.4329910, 225.4024048, -123.6448669, 237.1756439, -354.6085815, 349.0472717

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4289571, upper bound: 339.4307018
time: 1.02 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4289571, upper bound: 339.4310175
time: 0.64 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.24 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 4, lower bound: -339.4289571, upper bound: 339.4289571
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 4, lower bound: -339.4289571, upper bound: 339.4289571
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 4, lower bound: -339.4289571, upper bound: 339.4307018
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 4, lower bound: -339.4289571, upper bound: 339.4310175

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -119.5369644, 181.6595764, -119.5369644, 181.6595764, -301.1965027, 301.1965027
1: -95.3491898, 176.5832367, -95.3491898, 176.5832367, -271.9324341, 271.9324341
2: -82.4457779, 180.8220520, -82.4457779, 180.8220520, -263.2677917, 263.2677917
3: -124.2547455, 178.3655853, -124.2547455, 178.3655853, -302.6203308, 302.6203308
4: -99.1074524, 191.3053741, -99.1074524, 191.3053741, -290.4128418, 290.4128418

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4217991, upper bound: 339.4098882
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4070383, upper bound: 339.4070383
time: 0.87 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -119.5369644, 181.6595764, -142.2355347, 215.0777283, -334.6146851, 323.8950500
1: -95.3491898, 176.5832367, -113.0008011, 207.8044281, -303.1535339, 289.5840454
2: -82.4457779, 180.8220520, -97.5341110, 213.1814423, -295.6271362, 278.3561096
3: -124.2547455, 178.3655853, -147.2916718, 210.6535492, -334.9082947, 325.6572571
4: -99.1074524, 191.3053741, -117.4329910, 225.4024048, -324.5098572, 308.7383423

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4217991, upper bound: 339.4172764
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4070383, upper bound: 339.4144263
time: 0.97 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -142.2355347, 215.0777283, -119.5369644, 181.6595764, -323.8950500, 334.6146851
1: -113.0008011, 207.8044281, -95.3491898, 176.5832367, -289.5840454, 303.1535339
2: -97.5341110, 213.1814423, -82.4457779, 180.8220520, -278.3561096, 295.6271362
3: -147.2916718, 210.6535492, -124.2547455, 178.3655853, -325.6572571, 334.9082947
4: -117.4329910, 225.4024048, -99.1074524, 191.3053741, -308.7383423, 324.5098572

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4205131, upper bound: 339.4102888
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4144263, upper bound: 339.4098285
time: 1.03 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -142.2355347, 215.0777283, -142.2355347, 215.0777283, -357.3132629, 357.3132629
1: -113.0008011, 207.8044281, -113.0008011, 207.8044281, -320.8051758, 320.8051758
2: -97.5341110, 213.1814423, -97.5341110, 213.1814423, -310.7155151, 310.7155151
3: -147.2916718, 210.6535492, -147.2916718, 210.6535492, -357.9452209, 357.9452209
4: -117.4329910, 225.4024048, -117.4329910, 225.4024048, -342.8353882, 342.8353882

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4205131, upper bound: 339.4168689
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4144263, upper bound: 339.4168650
time: 0.92 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.45 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 4, lower bound: -339.4217991, upper bound: 339.4098882
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.45
Output dim: 4, lower bound: -339.4070383, upper bound: 339.4070383
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 4, lower bound: -339.4217991, upper bound: 339.4172764
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.45
Output dim: 4, lower bound: -339.4070383, upper bound: 339.4144263
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 4, lower bound: -339.4205131, upper bound: 339.4102888
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.45
Output dim: 4, lower bound: -339.4144263, upper bound: 339.4098285
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 4, lower bound: -339.4205131, upper bound: 339.4168689
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 4, lower bound: -339.4144263, upper bound: 339.4168650

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -110.0765305, 167.2630615, -119.5369644, 181.6595764, -291.7361145, 286.7999878
1: -87.7360077, 162.4888153, -95.3491898, 176.5832367, -264.3192444, 257.8379822
2: -75.8751373, 166.5349579, -82.4457779, 180.8220520, -256.6971741, 248.9807281
3: -114.4463348, 164.1009979, -124.2547455, 178.3655853, -292.8118896, 288.3557434
4: -91.2258987, 176.1902161, -99.1074524, 191.3053741, -282.5312805, 275.2976685

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4070383, upper bound: 339.4070383
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4070383, upper bound: 339.4070383
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -110.0765305, 167.2630615, -142.2355347, 215.0777283, -325.1542664, 309.4985657
1: -87.7360077, 162.4888153, -113.0008011, 207.8044281, -295.5403137, 275.4896240
2: -75.8751373, 166.5349579, -97.5341110, 213.1814423, -289.0565186, 264.0690613
3: -114.4463348, 164.1009979, -147.2916718, 210.6535492, -325.0998535, 311.3926697
4: -91.2258987, 176.1902161, -117.4329910, 225.4024048, -316.6282959, 293.6231689

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4098285, upper bound: 339.4144263
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4098285, upper bound: 339.4144263
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -132.1636047, 199.5604858, -119.5369644, 181.6595764, -313.8231506, 319.0974426
1: -104.9435425, 192.8429871, -95.3491898, 176.5832367, -281.5267944, 288.1921692
2: -90.5881958, 197.9390259, -82.4457779, 180.8220520, -271.4102173, 280.3847351
3: -136.9749603, 195.4505768, -124.2547455, 178.3655853, -315.3404846, 319.7053223
4: -109.0377426, 209.3752441, -99.1074524, 191.3053741, -300.3430786, 308.4826965

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4144263, upper bound: 339.4098285
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4144263, upper bound: 339.4098285
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -132.1636047, 199.5604858, -142.2355347, 215.0777283, -347.2413330, 341.7960205
1: -104.9435425, 192.8429871, -113.0008011, 207.8044281, -312.7478333, 305.8437805
2: -90.5881958, 197.9390259, -97.5341110, 213.1814423, -303.7696228, 295.4731140
3: -136.9749603, 195.4505768, -147.2916718, 210.6535492, -347.6284790, 342.7422485
4: -109.0377426, 209.3752441, -117.4329910, 225.4024048, -334.4401550, 326.8082275

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4172167, upper bound: 339.4168505
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4172167, upper bound: 339.4168505
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -151.6397247, 228.5525665, -140.7367706, 212.8262787, -364.4660034, 369.2892761
1: -120.1573105, 220.5421448, -111.8296280, 205.6829071, -325.8402100, 332.3717651
2: -103.8738251, 226.8395691, -96.5280914, 210.9666748, -314.8405151, 323.3676758
3: -157.4124451, 223.4964447, -145.7708282, 208.4755249, -365.8879700, 369.2672729
4: -124.9241486, 240.1870728, -116.2130127, 223.0550995, -347.9792480, 356.4000854

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4172167, upper bound: 339.4168650
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4172167, upper bound: 339.4168650
time: 0.96 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.40 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.40
Output dim: 4, lower bound: -339.4070383, upper bound: 339.4070383
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.40
Output dim: 4, lower bound: -339.4070383, upper bound: 339.4070383
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.40
Output dim: 4, lower bound: -339.4098285, upper bound: 339.4144263
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.40
Output dim: 4, lower bound: -339.4098285, upper bound: 339.4144263
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.40
Output dim: 4, lower bound: -339.4144263, upper bound: 339.4098285
NS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.40
Output dim: 4, lower bound: -339.4144263, upper bound: 339.4098285
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 4, lower bound: -339.4172167, upper bound: 339.4168505
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 4, lower bound: -339.4172167, upper bound: 339.4168505
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 4, lower bound: -339.4172167, upper bound: 339.4168650
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 4, lower bound: -339.4172167, upper bound: 339.4168650

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -132.1636047, 199.5604858, -132.1636047, 199.5604858, -331.7240906, 331.7240906
1: -104.9435425, 192.8429871, -104.9435425, 192.8429871, -297.7864990, 297.7864990
2: -90.5881958, 197.9390259, -90.5881958, 197.9390259, -288.5272217, 288.5272217
3: -136.9749603, 195.4505768, -136.9749603, 195.4505768, -332.4255066, 332.4255066
4: -109.0377426, 209.3752441, -109.0377426, 209.3752441, -318.4129944, 318.4129944

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4244902, upper bound: 339.4159616
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4244782, upper bound: 339.4161461
time: 1.03 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -132.1636047, 199.5604858, -151.6397247, 228.5525665, -360.7161255, 351.2001953
1: -104.9435425, 192.8429871, -120.1573105, 220.5421448, -325.4856873, 313.0003052
2: -90.5881958, 197.9390259, -103.8738251, 226.8395691, -317.4277649, 301.8128052
3: -136.9749603, 195.4505768, -157.4124451, 223.4964447, -360.4713135, 352.8630066
4: -109.0377426, 209.3752441, -124.9241486, 240.1870728, -349.2247925, 334.2993774

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4244902, upper bound: 339.4159616
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4244782, upper bound: 339.4161461
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -151.6397247, 228.5525665, -132.1636047, 199.5604858, -351.2001953, 360.7161255
1: -120.1573105, 220.5421448, -104.9435425, 192.8429871, -313.0003052, 325.4856873
2: -103.8738251, 226.8395691, -90.5881958, 197.9390259, -301.8128052, 317.4277649
3: -157.4124451, 223.4964447, -136.9749603, 195.4505768, -352.8630066, 360.4713440
4: -124.9241486, 240.1870728, -109.0377426, 209.3752441, -334.2993774, 349.2247925

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4115267, upper bound: 339.4104401
time: 1.18 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4172167, upper bound: 339.4168650
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -151.6397247, 228.5525665, -151.6397247, 228.5525665, -380.1922913, 380.1922913
1: -120.1573105, 220.5421448, -120.1573105, 220.5421448, -340.6994629, 340.6994629
2: -103.8738251, 226.8395691, -103.8738251, 226.8395691, -330.7133789, 330.7133789
3: -157.4124451, 223.4964447, -157.4124451, 223.4964447, -380.9088440, 380.9088440
4: -124.9241486, 240.1870728, -124.9241486, 240.1870728, -365.1112061, 365.1112061

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4115267, upper bound: 339.4104401
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4172167, upper bound: 339.4168650
time: 0.86 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.52 seconds
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.52
Output dim: 4, lower bound: -339.4244902, upper bound: 339.4159616
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.52
Output dim: 4, lower bound: -339.4244782, upper bound: 339.4161461
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.52
Output dim: 4, lower bound: -339.4244902, upper bound: 339.4159616
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.52
Output dim: 4, lower bound: -339.4244782, upper bound: 339.4161461
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.52
Output dim: 4, lower bound: -339.4115267, upper bound: 339.4104401
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.52
Output dim: 4, lower bound: -339.4172167, upper bound: 339.4168650
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.52
Output dim: 4, lower bound: -339.4115267, upper bound: 339.4104401
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.52
Output dim: 4, lower bound: -339.4172167, upper bound: 339.4168650

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -120.9359512, 182.4429932, -129.9253082, 196.1295776, -317.0655212, 312.3682861
1: -96.0795593, 176.3457336, -103.1725235, 189.5362396, -285.6157837, 279.5182495
2: -82.9002304, 181.0068207, -89.0513992, 194.5464478, -277.4466858, 270.0581665
3: -125.4588242, 178.7555847, -134.6762390, 192.1072845, -317.5661011, 313.4317017
4: -99.8061142, 191.5045471, -107.1941299, 205.7980194, -305.6041260, 298.6986694

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4109126, upper bound: 339.4144639
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4210278, upper bound: 339.4259001
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4210278, upper bound: 339.4259001
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -134.8745117, 203.0401306, -130.6416016, 197.2311249, -332.1056519, 333.6817322
1: -106.8430557, 195.7683716, -103.7320480, 190.5997772, -297.4428406, 299.5004272
2: -92.1843796, 201.0905914, -89.5402222, 195.6403351, -287.8247070, 290.6307983
3: -139.6303864, 198.5925751, -135.4130554, 193.1759949, -332.8063965, 334.0055542
4: -111.0596237, 212.7449188, -107.7815781, 206.9530487, -318.0126648, 320.5264587

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4156554, upper bound: 339.4217934
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4211967, upper bound: 339.4217957
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -120.9359512, 182.4429932, -149.6094360, 225.4266357, -346.3625793, 332.0523987
1: -96.0795593, 176.3457336, -118.5485535, 217.5174103, -313.5969849, 294.8942871
2: -82.9002304, 181.0068207, -102.4755554, 223.7507629, -306.6510010, 283.4823608
3: -125.4588242, 178.7555847, -155.3381500, 220.4417572, -345.9005737, 334.0937500
4: -99.8061142, 191.5045471, -123.2364883, 236.9375153, -336.7436218, 314.7410278

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4108336, upper bound: 339.4053585
time: 1.20 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4132237, upper bound: 339.4045079
time: 1.09 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -134.8745117, 203.0401306, -149.8927155, 225.9068146, -360.7813110, 352.9328003
1: -106.8430557, 195.7683716, -118.7668991, 217.9984894, -324.8415527, 314.5352783
2: -92.1843796, 201.0905914, -102.6707153, 224.2376862, -316.4220581, 303.7612915
3: -139.6303864, 198.5925751, -155.6144409, 220.9180756, -360.5484619, 354.2069397
4: -111.0596237, 212.7449188, -123.4787140, 237.4383392, -348.4979248, 336.2236328

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4102565, upper bound: 339.4085998
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4177858, upper bound: 339.4091013
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -148.1357117, 223.2130737, -132.1636047, 199.5604858, -347.6961975, 355.3766785
1: -117.3777237, 215.3220367, -104.9435425, 192.8429871, -310.2207031, 320.2655334
2: -101.4666290, 221.5437012, -90.5881958, 197.9390259, -299.4056396, 312.1318970
3: -153.8476257, 218.2608337, -136.9749603, 195.4505768, -349.2981873, 355.2357788
4: -122.0325851, 234.6056366, -109.0377426, 209.3752441, -331.4078369, 343.6433716

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4141668, upper bound: 339.4266730
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4141668, upper bound: 339.4266730
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -148.1357117, 223.2130737, -151.6397247, 228.5525665, -376.6882324, 374.8527832
1: -117.3777237, 215.3220367, -120.1573105, 220.5421448, -337.9198608, 335.4793396
2: -101.4666290, 221.5437012, -103.8738251, 226.8395691, -328.3062134, 325.4175415
3: -153.8476257, 218.2608337, -157.4124451, 223.4964447, -377.3440247, 375.6732788
4: -122.0325851, 234.6056366, -124.9241486, 240.1870728, -362.2196655, 359.5297852

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4059896, upper bound: 339.4080178
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4059896, upper bound: 339.4168650
time: 0.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.36 seconds
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -339.4210278, upper bound: 339.4259001
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -339.4210278, upper bound: 339.4259001
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -339.4156554, upper bound: 339.4217934
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -339.4211967, upper bound: 339.4217957
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.36
Output dim: 4, lower bound: -339.4108336, upper bound: 339.4053585
NS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.36
Output dim: 4, lower bound: -339.4132237, upper bound: 339.4045079
NS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.36
Output dim: 4, lower bound: -339.4102565, upper bound: 339.4085998
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -339.4177858, upper bound: 339.4091013
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -339.4141668, upper bound: 339.4266730
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -339.4141668, upper bound: 339.4266730
NS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.36
Output dim: 4, lower bound: -339.4059896, upper bound: 339.4080178
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -339.4059896, upper bound: 339.4168650

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -120.9359512, 182.4429932, -120.9359512, 182.4429932, -303.3789368, 303.3789368
1: -96.0795593, 176.3457336, -96.0795593, 176.3457336, -272.4252930, 272.4252930
2: -82.9002304, 181.0068207, -82.9002304, 181.0068207, -263.9070435, 263.9070435
3: -125.4588242, 178.7555847, -125.4588242, 178.7555847, -304.2144165, 304.2144165
4: -99.8061142, 191.5045471, -99.8061142, 191.5045471, -291.3106689, 291.3106689

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4242806, upper bound: 339.4221326
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4249753, upper bound: 339.4255353
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -120.9359512, 182.4429932, -134.8745117, 203.0401306, -323.9760437, 317.3175049
1: -96.0795593, 176.3457336, -106.8430557, 195.7683716, -291.8479309, 283.1887817
2: -82.9002304, 181.0068207, -92.1843796, 201.0905914, -283.9908142, 273.1911926
3: -125.4588242, 178.7555847, -139.6303864, 198.5925751, -324.0513916, 318.3859863
4: -99.8061142, 191.5045471, -111.0596237, 212.7449188, -312.5510254, 302.5641479

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4242806, upper bound: 339.4221326
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4249753, upper bound: 339.4255353
time: 1.03 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -134.8745117, 203.0401306, -125.5738831, 189.4879303, -324.3624268, 328.6139832
1: -106.8430557, 195.7683716, -99.7418060, 183.1690674, -290.0120544, 295.5101929
2: -92.1843796, 201.0905914, -86.0844727, 188.0176697, -280.2020569, 287.1750488
3: -139.6303864, 198.5925751, -130.2442017, 185.6167145, -325.2471008, 328.8367920
4: -111.0596237, 212.7449188, -103.6237564, 198.9215546, -309.9811401, 316.3686218

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4156554, upper bound: 339.4217893
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4156554, upper bound: 339.4217893
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -133.3272247, 200.7523041, -121.8212204, 184.3631134, -317.6903381, 322.5734863
1: -105.6541214, 193.5992279, -97.0427551, 178.4198608, -284.0739746, 290.6419678
2: -91.1588745, 198.8605347, -83.7864609, 183.0357666, -274.1946411, 282.6469421
3: -138.0882874, 196.3773804, -126.7007446, 180.7071686, -318.7954102, 323.0780945
4: -109.8195190, 210.3964233, -100.8155899, 193.6725922, -303.4920654, 311.2119446

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4211967, upper bound: 339.4217957
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4211967, upper bound: 339.4217957
time: 1.06 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -133.3272247, 200.7523041, -144.4167328, 218.0523834, -351.3796082, 345.1690369
1: -105.6541214, 193.5992279, -114.6766815, 210.7118225, -316.3659363, 308.2759094
2: -91.1588745, 198.8605347, -99.1572495, 216.6258240, -307.7846680, 298.0177612
3: -138.0882874, 196.3773804, -150.2834015, 213.3336029, -351.4218445, 346.6607666
4: -109.8195190, 210.3964233, -119.1906586, 229.4019165, -339.2214355, 329.5870667

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4177858, upper bound: 339.4091013
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4177858, upper bound: 339.4091013
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -148.1357117, 223.2130737, -123.5834503, 187.5995026, -335.7352295, 346.7964783
1: -117.3777237, 215.3220367, -98.2525711, 181.0865021, -298.4642029, 313.5746155
2: -101.4666290, 221.5437012, -84.7735291, 185.6113129, -287.0779419, 306.3172302
3: -153.8476257, 218.2608337, -127.7924957, 183.5115814, -337.3591614, 346.0533142
4: -122.0325851, 234.6056366, -102.0378799, 196.0400696, -318.0726624, 336.6435242

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4095174, upper bound: 339.4161873
time: 1.13 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4139727, upper bound: 339.4266023
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -148.1357117, 223.2130737, -129.1222229, 195.0547791, -343.1904907, 352.3352356
1: -117.3777237, 215.3220367, -102.5770264, 188.5191040, -305.8967896, 317.8990479
2: -101.4666290, 221.5437012, -88.5473328, 193.4957581, -294.9623718, 310.0910339
3: -153.8476257, 218.2608337, -133.9011230, 191.0614929, -344.9090576, 352.1619568
4: -122.0325851, 234.6056366, -106.5954132, 204.6688995, -326.7014465, 341.2010498

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4095174, upper bound: 339.4161873
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4139727, upper bound: 339.4266023
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -148.1357117, 223.2130737, -148.1357117, 223.2130737, -371.3487549, 371.3487549
1: -117.3777237, 215.3220367, -117.3777237, 215.3220367, -332.6997681, 332.6997681
2: -101.4666290, 221.5437012, -101.4666290, 221.5437012, -323.0103149, 323.0103149
3: -153.8476257, 218.2608337, -153.8476257, 218.2608337, -372.1084595, 372.1084595
4: -122.0325851, 234.6056366, -122.0325851, 234.6056366, -356.6382141, 356.6382141

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4052526, upper bound: 339.4071564
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4053759, upper bound: 339.4167002
time: 0.78 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.16 seconds
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 4, lower bound: -339.4242806, upper bound: 339.4221326
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 4, lower bound: -339.4249753, upper bound: 339.4255353
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 4, lower bound: -339.4242806, upper bound: 339.4221326
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 4, lower bound: -339.4249753, upper bound: 339.4255353
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 4, lower bound: -339.4156554, upper bound: 339.4217893
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 4, lower bound: -339.4156554, upper bound: 339.4217893
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 4, lower bound: -339.4211967, upper bound: 339.4217957
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 4, lower bound: -339.4211967, upper bound: 339.4217957
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 4, lower bound: -339.4177858, upper bound: 339.4091013
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 4, lower bound: -339.4177858, upper bound: 339.4091013
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 4, lower bound: -339.4095174, upper bound: 339.4161873
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 4, lower bound: -339.4139727, upper bound: 339.4266023
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 4, lower bound: -339.4095174, upper bound: 339.4161873
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 4, lower bound: -339.4139727, upper bound: 339.4266023
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.16
Output dim: 4, lower bound: -339.4052526, upper bound: 339.4071564
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 4, lower bound: -339.4053759, upper bound: 339.4167002

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -120.5757141, 182.1539459, -120.9359512, 182.4429932, -303.0187073, 303.0898743
1: -96.0279083, 175.9860840, -96.0795593, 176.3457336, -272.3736267, 272.0656433
2: -82.8228226, 180.6437073, -82.9002304, 181.0068207, -263.8296509, 263.5439453
3: -125.3043137, 178.5156860, -125.4588242, 178.7555847, -304.0599060, 303.9744873
4: -99.7458572, 191.0728912, -99.8061142, 191.5045471, -291.2503967, 290.8789978

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4235479, upper bound: 339.4235488
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4235479, upper bound: 339.4235488
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -118.6592178, 179.0278778, -120.9359512, 182.4429932, -301.1022034, 299.9638062
1: -94.2928772, 173.0048981, -96.0795593, 176.3457336, -270.6386108, 269.0844727
2: -81.3554382, 177.5861664, -82.9002304, 181.0068207, -262.3622131, 260.4863892
3: -123.1427536, 175.3806152, -125.4588242, 178.7555847, -301.8983154, 300.8394470
4: -97.9393539, 187.8928680, -99.8061142, 191.5045471, -289.4439087, 287.6989746

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4235479, upper bound: 339.4264157
time: 1.13 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4235479, upper bound: 339.4264157
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -120.5757141, 182.1539459, -134.8745117, 203.0401306, -323.6158142, 317.0284424
1: -96.0279083, 175.9860840, -106.8430557, 195.7683716, -291.7962646, 282.8291321
2: -82.8228226, 180.6437073, -92.1843796, 201.0905914, -283.9134216, 272.8280945
3: -125.3043137, 178.5156860, -139.6303864, 198.5925751, -323.8968811, 318.1460571
4: -99.7458572, 191.0728912, -111.0596237, 212.7449188, -312.4907532, 302.1325073

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4261905, upper bound: 339.4221326
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4261905, upper bound: 339.4221326
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -118.6592178, 179.0278778, -134.8745117, 203.0401306, -321.6993103, 313.9024048
1: -94.2928772, 173.0048981, -106.8430557, 195.7683716, -290.0612488, 279.8479614
2: -81.3554382, 177.5861664, -92.1843796, 201.0905914, -282.4460449, 269.7705383
3: -123.1427536, 175.3806152, -139.6303864, 198.5925751, -321.7353210, 315.0109863
4: -97.9393539, 187.8928680, -111.0596237, 212.7449188, -310.6842651, 298.9524536

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4261905, upper bound: 339.4255353
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4261905, upper bound: 339.4255353
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -131.1424866, 197.3173370, -125.5738831, 189.4879303, -320.6303711, 322.8912048
1: -103.8931274, 190.2356567, -99.7418060, 183.1690674, -287.0620728, 289.9773865
2: -89.6270142, 195.4125977, -86.0844727, 188.0176697, -277.6446838, 281.4970703
3: -135.8064575, 192.9751282, -130.2442017, 185.6167145, -321.4231567, 323.2193298
4: -107.9875641, 206.7593842, -103.6237564, 198.9215546, -306.9091187, 310.3830566

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4090375, upper bound: 339.4213089
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4090375, upper bound: 339.4217934
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -122.7522430, 185.4610443, -125.5738831, 189.4879303, -312.2400818, 311.0349121
1: -97.6838303, 179.1741180, -99.7418060, 183.1690674, -280.8528442, 278.9159241
2: -84.2895508, 183.9343719, -86.0844727, 188.0176697, -272.3072205, 270.0188293
3: -127.6990509, 181.6106262, -130.2442017, 185.6167145, -313.3157654, 311.8547974
4: -101.4892502, 194.6602478, -103.6237564, 198.9215546, -300.4107971, 298.2839661

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4090375, upper bound: 339.4213089
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4090375, upper bound: 339.4217934
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -131.1424866, 197.3173370, -121.8212204, 184.3631134, -315.5055847, 319.1384888
1: -103.8931274, 190.2356567, -97.0427551, 178.4198608, -282.3129272, 287.2783203
2: -89.6270142, 195.4125977, -83.7864609, 183.0357666, -272.6627502, 279.1990662
3: -135.8064575, 192.9751282, -126.7007446, 180.7071686, -316.5136108, 319.6758118
4: -107.9875641, 206.7593842, -100.8155899, 193.6725922, -301.6601562, 307.5748901

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4137810, upper bound: 339.4217548
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4137810, upper bound: 339.4140431
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -122.7522430, 185.4610443, -121.8212204, 184.3631134, -307.1153259, 307.2821655
1: -97.6838303, 179.1741180, -97.0427551, 178.4198608, -276.1036987, 276.2168579
2: -84.2895508, 183.9343719, -83.7864609, 183.0357666, -267.3252869, 267.7208252
3: -127.6990509, 181.6106262, -126.7007446, 180.7071686, -308.4061584, 308.3113403
4: -101.4892502, 194.6602478, -100.8155899, 193.6725922, -295.1618347, 295.4757996

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4137810, upper bound: 339.4217419
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4137810, upper bound: 339.4142903
time: 1.24 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -131.1424866, 197.3173370, -144.4167328, 218.0523834, -349.1948547, 341.7340393
1: -103.8931274, 190.2356567, -114.6766815, 210.7118225, -314.6049194, 304.9122620
2: -89.6270142, 195.4125977, -99.1572495, 216.6258240, -306.2528381, 294.5698547
3: -135.8064575, 192.9751282, -150.2834015, 213.3336029, -349.1400757, 343.2585449
4: -107.9875641, 206.7593842, -119.1906586, 229.4019165, -337.3894653, 325.9500122

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -122.7522430, 185.4610443, -144.4167328, 218.0523834, -340.8045959, 329.8777466
1: -97.6838303, 179.1741180, -114.6766815, 210.7118225, -308.3956604, 293.8507996
2: -84.2895508, 183.9343719, -99.1572495, 216.6258240, -300.9153748, 283.0916138
3: -127.6990509, 181.6106262, -150.2834015, 213.3336029, -341.0325623, 331.8940430
4: -101.4892502, 194.6602478, -119.1906586, 229.4019165, -330.8911743, 313.8508911

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -146.9705353, 221.4877625, -123.5834503, 187.5995026, -334.5700378, 345.0711975
1: -116.4668808, 213.6318359, -98.2525711, 181.0865021, -297.5533752, 311.8843994
2: -100.6750565, 219.8135529, -84.7735291, 185.6113129, -286.2862854, 304.5870972
3: -152.6674500, 216.5588074, -127.7924957, 183.5115814, -336.1790161, 344.3512878
4: -121.0812912, 232.7727051, -102.0378799, 196.0400696, -317.1213379, 334.8105774

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4094308, upper bound: 339.4213152
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4086221, upper bound: 339.4165144
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -146.9705353, 221.4877625, -129.1222229, 195.0547791, -342.0253296, 350.6099548
1: -116.4668808, 213.6318359, -102.5770264, 188.5191040, -304.9859924, 316.2087708
2: -100.6750565, 219.8135529, -88.5473328, 193.4957581, -294.1707153, 308.3608704
3: -152.6674500, 216.5588074, -133.9011230, 191.0614929, -343.7289429, 350.4598999
4: -121.0812912, 232.7727051, -106.5954132, 204.6688995, -325.7501221, 339.3681030

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4091594, upper bound: 339.4137362
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4091594, upper bound: 339.4266023
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -146.9705353, 221.4877625, -148.1357117, 223.2130737, -370.1835938, 369.6234741
1: -116.4668808, 213.6318359, -117.3777237, 215.3220367, -331.7889099, 331.0094910
2: -100.6750565, 219.8135529, -101.4666290, 221.5437012, -322.2187500, 321.2801819
3: -152.6674500, 216.5588074, -153.8476257, 218.2608337, -370.9282837, 370.4064331
4: -121.0812912, 232.7727051, -122.0325851, 234.6056366, -355.6869202, 354.8052979

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4061623, upper bound: 339.4078352
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4061623, upper bound: 339.4167002
time: 0.79 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 3.37 seconds
NS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4235479, upper bound: 339.4235488
NS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4235479, upper bound: 339.4235488
NS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4235479, upper bound: 339.4264157
NS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4235479, upper bound: 339.4264157
NS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4261905, upper bound: 339.4221326
NS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4261905, upper bound: 339.4221326
NS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4261905, upper bound: 339.4255353
NS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4261905, upper bound: 339.4255353
NS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4090375, upper bound: 339.4213089
NS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4090375, upper bound: 339.4217934
NS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4090375, upper bound: 339.4213089
NS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4090375, upper bound: 339.4217934
NS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4137810, upper bound: 339.4217548
NS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4137810, upper bound: 339.4140431
NS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4137810, upper bound: 339.4217419
NS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4137810, upper bound: 339.4142903
NS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4094308, upper bound: 339.4213152
NS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4086221, upper bound: 339.4165144
NS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4091594, upper bound: 339.4137362
NS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4091594, upper bound: 339.4266023
NS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4061623, upper bound: 339.4078352
NS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.37
Output dim: 4, lower bound: -339.4061623, upper bound: 339.4167002

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -120.5757141, 182.1539459, -120.5757141, 182.1539459, -302.7296753, 302.7296753
1: -96.0279083, 175.9860840, -96.0279083, 175.9860840, -272.0139771, 272.0139771
2: -82.8228226, 180.6437073, -82.8228226, 180.6437073, -263.4665222, 263.4665222
3: -125.3043137, 178.5156860, -125.3043137, 178.5156860, -303.8199463, 303.8199463
4: -99.7458572, 191.0728912, -99.7458572, 191.0728912, -290.8187561, 290.8187561

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4229340, upper bound: 339.4135774
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4234145, upper bound: 339.4233141
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -120.5757141, 182.1539459, -118.6592178, 179.0278778, -299.6035767, 300.8131409
1: -96.0279083, 175.9860840, -94.2928772, 173.0048981, -269.0327148, 270.2789612
2: -82.8228226, 180.6437073, -81.3554382, 177.5861664, -260.4089661, 261.9991150
3: -125.3043137, 178.5156860, -123.1427536, 175.3806152, -300.6849365, 301.6583862
4: -99.7458572, 191.0728912, -97.9393539, 187.8928680, -287.6387024, 289.0122375

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4229339, upper bound: 339.4135774
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4234145, upper bound: 339.4233142
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -118.6592178, 179.0278778, -120.5757141, 182.1539459, -300.8131409, 299.6035767
1: -94.2928772, 173.0048981, -96.0279083, 175.9860840, -270.2789612, 269.0327148
2: -81.3554382, 177.5861664, -82.8228226, 180.6437073, -261.9991455, 260.4089966
3: -123.1427536, 175.3806152, -125.3043137, 178.5156860, -301.6583862, 300.6849365
4: -97.9393539, 187.8928680, -99.7458572, 191.0728912, -289.0122375, 287.6387024

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4231800, upper bound: 339.4209324
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4233305, upper bound: 339.4259880
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -118.6592178, 179.0278778, -118.6592178, 179.0278778, -297.6870728, 297.6870728
1: -94.2928772, 173.0048981, -94.2928772, 173.0048981, -267.2976990, 267.2976990
2: -81.3554382, 177.5861664, -81.3554382, 177.5861664, -258.9415283, 258.9415588
3: -123.1427536, 175.3806152, -123.1427536, 175.3806152, -298.5233765, 298.5233765
4: -97.9393539, 187.8928680, -97.9393539, 187.8928680, -285.8322144, 285.8322144

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4231799, upper bound: 339.4211158
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4233305, upper bound: 339.4259880
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -120.5757141, 182.1539459, -134.8069000, 203.3424683, -323.9181824, 316.9608154
1: -96.0279083, 175.9860840, -107.0597458, 196.0057678, -292.0336609, 283.0458374
2: -82.8228226, 180.6437073, -92.3342056, 201.3210144, -284.1437988, 272.9779053
3: -125.3043137, 178.5156860, -139.7855377, 198.9611664, -324.2654724, 318.3012085
4: -99.7458572, 191.0728912, -111.2841644, 212.9085999, -312.6544495, 302.3570557

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4262142, upper bound: 339.4161534
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4226444, upper bound: 339.4122784
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4262124, upper bound: 339.4220966
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -120.5757141, 182.1539459, -132.6107330, 199.6429596, -320.2186890, 314.7646790
1: -96.0279083, 175.9860840, -105.0620499, 192.4389038, -288.4667969, 281.0481262
2: -82.8228226, 180.6437073, -90.6436310, 197.6852875, -280.5080872, 271.2873535
3: -125.3043137, 178.5156860, -137.3243866, 195.2326508, -320.5369568, 315.8399963
4: -99.7458572, 191.0728912, -109.1989975, 209.1503448, -308.8961487, 300.2718811

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4262142, upper bound: 339.4162373
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4226444, upper bound: 339.4122784
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4262124, upper bound: 339.4220966
time: 1.19 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -118.6592178, 179.0278778, -134.8069000, 203.3424683, -322.0016785, 313.8347473
1: -94.2928772, 173.0048981, -107.0597458, 196.0057678, -290.2986145, 280.0646057
2: -81.3554382, 177.5861664, -92.3342056, 201.3210144, -282.6763916, 269.9203491
3: -123.1427536, 175.3806152, -139.7855377, 198.9611664, -322.1038818, 315.1661377
4: -97.9393539, 187.8928680, -111.2841644, 212.9085999, -310.8479614, 299.1770325

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4261905, upper bound: 339.4196022
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4213299, upper bound: 339.4196022
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -118.6592178, 179.0278778, -132.6107330, 199.6429596, -318.3021545, 311.6386108
1: -94.2928772, 173.0048981, -105.0620499, 192.4389038, -286.7317810, 278.0669556
2: -81.3554382, 177.5861664, -90.6436310, 197.6852875, -279.0407104, 268.2297363
3: -123.1427536, 175.3806152, -137.3243866, 195.2326508, -318.3753967, 312.7049561
4: -97.9393539, 187.8928680, -109.1989975, 209.1503448, -307.0896606, 297.0918579

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4242545, upper bound: 339.4199141
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4261867, upper bound: 339.4252027
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -131.1424866, 197.3173370, -116.1646271, 175.1411285, -306.2835693, 313.4819336
1: -103.8931274, 190.2356567, -92.3251343, 169.3117981, -273.2049255, 282.5606995
2: -89.6270142, 195.4125977, -79.6448746, 173.7966309, -263.4236145, 275.0574646
3: -135.8064575, 192.9751282, -120.5868759, 171.6058350, -307.4122925, 313.5620117
4: -107.9875641, 206.7593842, -95.8872299, 183.9049377, -291.8924255, 302.6465454

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4165691, upper bound: 339.4218046
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4170976, upper bound: 339.4227968
time: 1.07 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -131.1424866, 197.3173370, -131.1424866, 197.3173370, -328.4597473, 328.4597473
1: -103.8931274, 190.2356567, -103.8931274, 190.2356567, -294.1286926, 294.1286926
2: -89.6270142, 195.4125977, -89.6270142, 195.4125977, -285.0396118, 285.0396118
3: -135.8064575, 192.9751282, -135.8064575, 192.9751282, -328.7815857, 328.7815857
4: -107.9875641, 206.7593842, -107.9875641, 206.7593842, -314.7469177, 314.7469177

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4165691, upper bound: 339.4218250
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4170976, upper bound: 339.4227968
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -122.7522430, 185.4610443, -116.1646271, 175.1411285, -297.8933105, 301.6256409
1: -97.6838303, 179.1741180, -92.3251343, 169.3117981, -266.9956360, 271.4992371
2: -84.2895508, 183.9343719, -79.6448746, 173.7966309, -258.0861511, 263.5792236
3: -127.6990509, 181.6106262, -120.5868759, 171.6058350, -299.3048401, 302.1975098
4: -101.4892502, 194.6602478, -95.8872299, 183.9049377, -285.3941345, 290.5474548

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4076231, upper bound: 339.4211830
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4062056, upper bound: 339.4197208
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -122.7522430, 185.4610443, -131.1424866, 197.3173370, -320.0695190, 316.6034546
1: -97.6838303, 179.1741180, -103.8931274, 190.2356567, -287.9194336, 283.0672302
2: -84.2895508, 183.9343719, -89.6270142, 195.4125977, -279.7021484, 273.5614014
3: -127.6990509, 181.6106262, -135.8064575, 192.9751282, -320.6741333, 317.4170837
4: -101.4892502, 194.6602478, -107.9875641, 206.7593842, -308.2486267, 302.6478271

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4076231, upper bound: 339.4215156
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4062056, upper bound: 339.4215079
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -131.1424866, 197.3173370, -115.3984604, 174.4965515, -305.6390381, 312.7157593
1: -103.8931274, 190.2356567, -91.9243088, 168.7164917, -272.6096191, 282.1598511
2: -89.6270142, 195.4125977, -79.3609467, 173.1958923, -262.8229065, 274.7735596
3: -135.8064575, 192.9751282, -120.1037674, 170.9513397, -306.7578125, 313.0788879
4: -107.9875641, 206.7593842, -95.4744110, 183.3068237, -291.2943726, 302.2337341

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4140295, upper bound: 339.4145163
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4140295, upper bound: 339.4145163
time: 1.19 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -122.7522430, 185.4610443, -115.3984604, 174.4965515, -297.2487793, 300.8594666
1: -97.6838303, 179.1741180, -91.9243088, 168.7164917, -266.4003296, 271.0984192
2: -84.2895508, 183.9343719, -79.3609467, 173.1958923, -257.4854431, 263.2952881
3: -127.6990509, 181.6106262, -120.1037674, 170.9513397, -298.6503296, 301.7143860
4: -101.4892502, 194.6602478, -95.4744110, 183.3068237, -284.7960815, 290.1346436

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4138976, upper bound: 339.4204797
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4109216, upper bound: 339.4202581
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -146.9705353, 221.4877625, -115.2453003, 175.3063507, -322.2768860, 336.7330322
1: -116.4668808, 213.6318359, -91.7511826, 169.6627808, -286.1296692, 305.3829956
2: -100.6750565, 219.8135529, -79.1958618, 173.7511749, -274.4261475, 299.0093994
3: -152.6674500, 216.5588074, -119.2806396, 171.7988434, -324.4663086, 335.8394470
4: -121.0812912, 232.7727051, -95.2828522, 183.4892120, -304.5704956, 328.0555115

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4064789, upper bound: 339.4140263
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4085294, upper bound: 339.4207969
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -145.7300720, 219.6252289, -116.0566025, 177.0360107, -322.7660828, 335.6818237
1: -115.4757614, 211.8577576, -92.3454742, 171.0145721, -286.4902954, 304.2032166
2: -99.8164368, 217.9604797, -79.6732178, 175.0130615, -274.8294983, 297.6336975
3: -151.3539886, 214.7586670, -119.7935944, 173.2109375, -324.5649109, 334.5522461
4: -120.0475159, 230.8017120, -95.9225159, 184.5896606, -304.6371155, 326.7242432

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4057619, upper bound: 339.4139415
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4077356, upper bound: 339.4158295
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -146.9705353, 221.4877625, -128.2220154, 193.7435608, -340.7141113, 349.7097473
1: -116.4668808, 213.6318359, -101.8790588, 187.2569122, -303.7237854, 315.5108337
2: -100.6750565, 219.8135529, -87.9419327, 192.1886902, -292.8637390, 307.7554932
3: -152.6674500, 216.5588074, -132.9871063, 189.7785339, -342.4459839, 349.5458984
4: -121.0812912, 232.7727051, -105.8677216, 203.2804871, -324.3617554, 338.6404419

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4056633, upper bound: 339.4164215
time: 1.19 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4082595, upper bound: 339.4263045
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -146.9705353, 221.4877625, -146.9705353, 221.4877625, -368.4583130, 368.4583130
1: -116.4668808, 213.6318359, -116.4668808, 213.6318359, -330.0986938, 330.0986938
2: -100.6750565, 219.8135529, -100.6750565, 219.8135529, -320.4885559, 320.4885559
3: -152.6674500, 216.5588074, -152.6674500, 216.5588074, -369.2262573, 369.2262573
4: -121.0812912, 232.7727051, -121.0812912, 232.7727051, -353.8540039, 353.8540039

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4036323, upper bound: 339.4116041
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4051867, upper bound: 339.4162371
time: 0.99 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 3.50 seconds
NS_A2_B2_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4229340, upper bound: 339.4135774
NS_A2_B2_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4234145, upper bound: 339.4233141
NS_A2_B2_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4229339, upper bound: 339.4135774
NS_A2_B2_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4234145, upper bound: 339.4233142
NS_A2_B2_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4231800, upper bound: 339.4209324
NS_A2_B2_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4233305, upper bound: 339.4259880
NS_A2_B2_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4231799, upper bound: 339.4211158
NS_A2_B2_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4233305, upper bound: 339.4259880
NS_A2_B2_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4226444, upper bound: 339.4122784
NS_A2_B2_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4262124, upper bound: 339.4220966
NS_A2_B2_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4226444, upper bound: 339.4122784
NS_A2_B2_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4262124, upper bound: 339.4220966
NS_A2_B2_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4261905, upper bound: 339.4196022
NS_A2_B2_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4213299, upper bound: 339.4196022
NS_A2_B2_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4242545, upper bound: 339.4199141
NS_A2_B2_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4261867, upper bound: 339.4252027
NS_A2_B2_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4165691, upper bound: 339.4218046
NS_A2_B2_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4170976, upper bound: 339.4227968
NS_A2_B2_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4165691, upper bound: 339.4218250
NS_A2_B2_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4170976, upper bound: 339.4227968
NS_A2_B2_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4076231, upper bound: 339.4211830
NS_A2_B2_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4062056, upper bound: 339.4197208
NS_A2_B2_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4076231, upper bound: 339.4215156
NS_A2_B2_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4062056, upper bound: 339.4215079
NS_A2_B2_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4140295, upper bound: 339.4145163
NS_A2_B2_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4140295, upper bound: 339.4145163
NS_A2_B2_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4138976, upper bound: 339.4204797
NS_A2_B2_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4109216, upper bound: 339.4202581
NS_A2_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4064789, upper bound: 339.4140263
NS_A2_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4085294, upper bound: 339.4207969
NS_A2_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4057619, upper bound: 339.4139415
NS_A2_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4077356, upper bound: 339.4158295
NS_A2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4056633, upper bound: 339.4164215
NS_A2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4082595, upper bound: 339.4263045
NS_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4036323, upper bound: 339.4116041
NS_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.50
Output dim: 4, lower bound: -339.4051867, upper bound: 339.4162371

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -122.5723801, 185.2850342, -120.5757141, 182.1539459, -304.7263184, 305.8606873
1: -97.7203217, 179.0567474, -96.0279083, 175.9860840, -273.7064209, 275.0845642
2: -84.2710648, 183.7273407, -82.8228226, 180.6437073, -264.9147644, 266.5501709
3: -127.4514313, 181.7063293, -125.3043137, 178.5156860, -305.9671021, 307.0106201
4: -101.4752197, 194.3064423, -99.7458572, 191.0728912, -292.5480957, 294.0522461

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4135961, upper bound: 339.4139671
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4135961, upper bound: 339.4140459
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -119.3333206, 180.3211670, -120.5757141, 182.1539459, -301.4872437, 300.8968811
1: -95.0598526, 174.2033386, -96.0279083, 175.9860840, -271.0459290, 270.2312622
2: -81.9842834, 178.8122864, -82.8228226, 180.6437073, -262.6279907, 261.6351013
3: -124.0454788, 176.7102661, -125.3043137, 178.5156860, -302.5611572, 302.0145874
4: -98.7374420, 189.1328430, -99.7458572, 191.0728912, -289.8103333, 288.8786621

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4062717, upper bound: 339.4233586
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4062717, upper bound: 339.4236891
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -122.5723801, 185.2850342, -118.6592178, 179.0278778, -301.6002502, 303.9442139
1: -97.7203217, 179.0567474, -94.2928772, 173.0048981, -270.7252197, 273.3495483
2: -84.2710648, 183.7273407, -81.3554382, 177.5861664, -261.8572083, 265.0827637
3: -127.4514313, 181.7063293, -123.1427536, 175.3806152, -302.8320312, 304.8490295
4: -101.4752197, 194.3064423, -97.9393539, 187.8928680, -289.3680420, 292.2457581

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4139285, upper bound: 339.4135774
time: 1.19 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4139286, upper bound: 339.4135774
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -119.3333206, 180.3211670, -118.6592178, 179.0278778, -298.3611755, 298.9803772
1: -95.0598526, 174.2033386, -94.2928772, 173.0048981, -268.0646973, 268.4962158
2: -81.9842834, 178.8122864, -81.3554382, 177.5861664, -259.5704346, 260.1676636
3: -124.0454788, 176.7102661, -123.1427536, 175.3806152, -299.4260864, 299.8530273
4: -98.7374420, 189.1328430, -97.9393539, 187.8928680, -286.6303101, 287.0721741

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4050551, upper bound: 339.4232740
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4139286, upper bound: 339.4233141
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -120.6980896, 182.1997681, -120.5757141, 182.1539459, -302.8520203, 302.7754517
1: -96.0438080, 176.1367798, -96.0279083, 175.9860840, -272.0299072, 272.1646118
2: -82.8602142, 180.7209473, -82.8228226, 180.6437073, -263.5039062, 263.5437622
3: -125.3560944, 178.6244965, -125.3043137, 178.5156860, -303.8717346, 303.9288025
4: -99.7380981, 191.1845703, -99.7458572, 191.0728912, -290.8109741, 290.9304199

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4135774, upper bound: 339.4208297
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4135774, upper bound: 339.4209324
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -117.5599213, 177.4074860, -120.5757141, 182.1539459, -299.7138672, 297.9832153
1: -93.4347382, 171.4351501, -96.0279083, 175.9860840, -269.4208374, 267.4629822
2: -80.6119690, 175.9689789, -82.8228226, 180.6437073, -261.2556763, 258.7917786
3: -122.0255203, 173.7874908, -125.3043137, 178.5156860, -300.5411682, 299.0917053
4: -97.0450668, 186.1791840, -99.7458572, 191.0728912, -288.1179504, 285.9250488

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4135774, upper bound: 339.4249188
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4135774, upper bound: 339.4262130
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -120.6980896, 182.1997681, -118.6592178, 179.0278778, -299.7259521, 300.8589478
1: -96.0438080, 176.1367798, -94.2928772, 173.0048981, -269.0486755, 270.4295959
2: -82.8602142, 180.7209473, -81.3554382, 177.5861664, -260.4463806, 262.0763245
3: -125.3560944, 178.6244965, -123.1427536, 175.3806152, -300.7366943, 301.7672119
4: -99.7380981, 191.1845703, -97.9393539, 187.8928680, -287.6309814, 289.1239319

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4168293, upper bound: 339.4211158
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4168293, upper bound: 339.4211158
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -117.5599213, 177.4074860, -118.6592178, 179.0278778, -296.5877991, 296.0667114
1: -93.4347382, 171.4351501, -94.2928772, 173.0048981, -266.4396057, 265.7279663
2: -80.6119690, 175.9689789, -81.3554382, 177.5861664, -258.1981201, 257.3243713
3: -122.0255203, 173.7874908, -123.1427536, 175.3806152, -297.4061279, 296.9301453
4: -97.0450668, 186.1791840, -97.9393539, 187.8928680, -284.9378662, 284.1185303

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4034707, upper bound: 339.4251675
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4168293, upper bound: 339.4259880
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -122.5723801, 185.2850342, -134.8069000, 203.3424683, -325.9148560, 320.0918884
1: -97.7203217, 179.0567474, -107.0597458, 196.0057678, -293.7260742, 286.1164551
2: -84.2710648, 183.7273407, -92.3342056, 201.3210144, -285.5920105, 276.0615540
3: -127.4514313, 181.7063293, -139.7855377, 198.9611664, -326.4125977, 321.4918823
4: -101.4752197, 194.3064423, -111.2841644, 212.9085999, -314.3837891, 305.5905457

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -119.3333206, 180.3211670, -134.8069000, 203.3424683, -322.6757812, 315.1280518
1: -95.0598526, 174.2033386, -107.0597458, 196.0057678, -291.0656128, 281.2630920
2: -81.9842834, 178.8122864, -92.3342056, 201.3210144, -283.3052673, 271.1464539
3: -124.0454788, 176.7102661, -139.7855377, 198.9611664, -323.0066528, 316.4957886
4: -98.7374420, 189.1328430, -111.2841644, 212.9085999, -311.6460571, 300.4169922

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -122.5723801, 185.2850342, -132.6107330, 199.6429596, -322.2153320, 317.8957520
1: -97.7203217, 179.0567474, -105.0620499, 192.4389038, -290.1592407, 284.1188049
2: -84.2710648, 183.7273407, -90.6436310, 197.6852875, -281.9563293, 274.3709717
3: -127.4514313, 181.7063293, -137.3243866, 195.2326508, -322.6840820, 319.0306702
4: -101.4752197, 194.3064423, -109.1989975, 209.1503448, -310.6254883, 303.5054321

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -119.3333206, 180.3211670, -132.6107330, 199.6429596, -318.9762573, 312.9318848
1: -95.0598526, 174.2033386, -105.0620499, 192.4389038, -287.4987488, 279.2653809
2: -81.9842834, 178.8122864, -90.6436310, 197.6852875, -279.6695557, 269.4558716
3: -124.0454788, 176.7102661, -137.3243866, 195.2326508, -319.2781372, 314.0346680
4: -98.7374420, 189.1328430, -109.1989975, 209.1503448, -307.8877258, 298.3318481

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -109.4973755, 164.9560242, -134.8069000, 203.3424683, -312.8398438, 299.7629395
1: -87.0287170, 159.2531433, -107.0597458, 196.0057678, -283.0344849, 266.3128662
2: -75.0659561, 163.6335602, -92.3342056, 201.3210144, -276.3868408, 255.9677277
3: -113.8193054, 161.5401459, -139.7855377, 198.9611664, -312.7804565, 301.3256836
4: -90.3632507, 173.2369843, -111.2841644, 212.9085999, -303.2718201, 284.5211487

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4209257, upper bound: 339.4196022
time: 1.17 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4209257, upper bound: 339.4196022
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -118.5599899, 177.7195282, -134.8069000, 203.3424683, -321.9024658, 312.5264282
1: -93.7300568, 171.4063721, -107.0597458, 196.0057678, -289.7357483, 278.4660645
2: -80.9339523, 176.3727417, -92.3342056, 201.3210144, -282.2549744, 268.7069397
3: -122.6427002, 173.9042206, -139.7855377, 198.9611664, -321.6038818, 313.6897583
4: -97.4791794, 186.7216034, -111.2841644, 212.9085999, -310.3877869, 298.0057678

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4209257, upper bound: 339.4196022
time: 1.14 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4209257, upper bound: 339.4196022
time: 1.07 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -120.6980896, 182.1997681, -132.6107330, 199.6429596, -320.3410645, 314.8104858
1: -96.0438080, 176.1367798, -105.0620499, 192.4389038, -288.4827271, 281.1987915
2: -82.8602142, 180.7209473, -90.6436310, 197.6852875, -280.5455017, 271.3645325
3: -125.3560944, 178.6244965, -137.3243866, 195.2326508, -320.5887451, 315.9488525
4: -99.7380981, 191.1845703, -109.1989975, 209.1503448, -308.8884277, 300.3835754

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4166199, upper bound: 339.4166611
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -117.5599213, 177.4074860, -132.6107330, 199.6429596, -317.2028809, 310.0182190
1: -93.4347382, 171.4351501, -105.0620499, 192.4389038, -285.8736572, 276.4971924
2: -80.6119690, 175.9689789, -90.6436310, 197.6852875, -278.2972412, 266.6125183
3: -122.0255203, 173.7874908, -137.3243866, 195.2326508, -317.2581787, 311.1117554
4: -97.0450668, 186.1791840, -109.1989975, 209.1503448, -306.1952820, 295.3781738

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4174544, upper bound: 339.4180640
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -131.1416321, 197.7618866, -116.1646271, 175.1411285, -306.2827148, 313.9264832
1: -104.1881714, 190.6064148, -92.3251343, 169.3117981, -273.4999695, 282.9314880
2: -89.8430481, 195.7831879, -79.6448746, 173.7966309, -263.6396790, 275.4279785
3: -136.0678253, 193.4932709, -120.5868759, 171.6058350, -307.6736450, 314.0801086
4: -108.2924194, 207.0696869, -95.8872299, 183.9049377, -292.1972656, 302.9569092

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4164824, upper bound: 339.4218053
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4164824, upper bound: 339.4218053
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -129.0135498, 194.1192627, -116.1646271, 175.1411285, -304.1546631, 310.2838745
1: -102.2144470, 187.0906677, -92.3251343, 169.3117981, -271.5262146, 279.4158020
2: -88.1730270, 192.1977844, -79.6448746, 173.7966309, -261.9695740, 271.8426514
3: -133.6322632, 189.8048401, -120.5868759, 171.6058350, -305.2380676, 310.3916931
4: -106.2310257, 203.3643646, -95.8872299, 183.9049377, -290.1358337, 299.2515869

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4165926, upper bound: 339.4230750
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4165926, upper bound: 339.4230750
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -131.1416321, 197.7618866, -131.1424866, 197.3173370, -328.4589233, 328.9042969
1: -104.1881714, 190.6064148, -103.8931274, 190.2356567, -294.4238281, 294.4994812
2: -89.8430481, 195.7831879, -89.6270142, 195.4125977, -285.2556458, 285.4101257
3: -136.0678253, 193.4932709, -135.8064575, 192.9751282, -329.0429077, 329.2997131
4: -108.2924194, 207.0696869, -107.9875641, 206.7593842, -315.0517578, 315.0572510

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4169142, upper bound: 339.4218250
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4169142, upper bound: 339.4218250
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -129.0135498, 194.1192627, -131.1424866, 197.3173370, -326.3308411, 325.2617493
1: -102.2144470, 187.0906677, -103.8931274, 190.2356567, -292.4499817, 290.9837952
2: -88.1730270, 192.1977844, -89.6270142, 195.4125977, -283.5856323, 281.8247986
3: -133.6322632, 189.8048401, -135.8064575, 192.9751282, -326.6073303, 325.6112976
4: -106.2310257, 203.3643646, -107.9875641, 206.7593842, -312.9903259, 311.3519287

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4169142, upper bound: 339.4227968
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4169142, upper bound: 339.4227968
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -125.9725113, 190.6332703, -116.1646271, 175.1411285, -301.1136475, 306.7979126
1: -100.4419861, 184.2600555, -92.3251343, 169.3117981, -269.7537842, 276.5852051
2: -86.6264496, 189.0677490, -79.6448746, 173.7966309, -260.4230652, 268.7125854
3: -131.1382141, 186.7916260, -120.5868759, 171.6058350, -302.7440186, 307.3785095
4: -104.3300552, 199.9936523, -95.8872299, 183.9049377, -288.2348633, 295.8808899

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4062056, upper bound: 339.4197208
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4062056, upper bound: 339.4197208
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -120.5760040, 182.2086029, -116.1646271, 175.1411285, -295.7171326, 298.3732300
1: -95.9774475, 176.0157166, -92.3251343, 169.3117981, -265.2892456, 268.3408203
2: -82.8138809, 180.6908264, -79.6448746, 173.7966309, -256.6104736, 260.3356934
3: -125.4860306, 178.4080658, -120.5868759, 171.6058350, -297.0917969, 298.9949036
4: -99.7066574, 191.2338104, -95.8872299, 183.9049377, -283.6114502, 287.1210327

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4062056, upper bound: 339.4197208
time: 1.11 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4062056, upper bound: 339.4197208
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -125.9725113, 190.6332703, -131.1424866, 197.3173370, -323.2898254, 321.7757568
1: -100.4419861, 184.2600555, -103.8931274, 190.2356567, -290.6776428, 288.1531677
2: -86.6264496, 189.0677490, -89.6270142, 195.4125977, -282.0390625, 278.6947632
3: -131.1382141, 186.7916260, -135.8064575, 192.9751282, -324.1133118, 322.5980835
4: -104.3300552, 199.9936523, -107.9875641, 206.7593842, -311.0893250, 307.9812012

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4137689, upper bound: 339.4213627
time: 1.09 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4137519, upper bound: 339.4137157
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -120.5760040, 182.2086029, -131.1424866, 197.3173370, -317.8933411, 313.3510437
1: -95.9774475, 176.0157166, -103.8931274, 190.2356567, -286.2130127, 279.9088135
2: -82.8138809, 180.6908264, -89.6270142, 195.4125977, -278.2264709, 270.3178406
3: -125.4860306, 178.4080658, -135.8064575, 192.9751282, -318.4610901, 314.2145081
4: -99.7066574, 191.2338104, -107.9875641, 206.7593842, -306.4659424, 299.2213745

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4137689, upper bound: 339.4213951
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4137583, upper bound: 339.4137524
time: 1.09 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -125.9725113, 190.6332703, -115.3984604, 174.4965515, -300.4690552, 306.0317383
1: -100.4419861, 184.2600555, -91.9243088, 168.7164917, -269.1584778, 276.1843567
2: -86.6264496, 189.0677490, -79.3609467, 173.1958923, -259.8223267, 268.4286499
3: -131.1382141, 186.7916260, -120.1037674, 170.9513397, -302.0895081, 306.8953857
4: -104.3300552, 199.9936523, -95.4744110, 183.3068237, -287.6367493, 295.4680481

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4138078, upper bound: 339.4202581
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4138078, upper bound: 339.4202581
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -120.5760040, 182.2086029, -115.3984604, 174.4965515, -295.0725708, 297.6070251
1: -95.9774475, 176.0157166, -91.9243088, 168.7164917, -264.6939392, 267.9400024
2: -82.8138809, 180.6908264, -79.3609467, 173.1958923, -256.0097656, 260.0517578
3: -125.4860306, 178.4080658, -120.1037674, 170.9513397, -296.4373169, 298.5117798
4: -99.7066574, 191.2338104, -95.4744110, 183.3068237, -283.0133667, 286.7082214

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4138078, upper bound: 339.4202581
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4138078, upper bound: 339.4202581
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -146.0002441, 220.0361328, -115.2453003, 175.3063507, -321.3065796, 335.2814331
1: -115.6960220, 212.2665558, -91.7511826, 169.6627808, -285.3587952, 304.0177307
2: -100.0173492, 218.4105225, -79.1958618, 173.7511749, -273.7684631, 297.6063843
3: -151.6720581, 215.1582336, -119.2806396, 171.7988434, -323.4708862, 334.4388733
4: -120.2893677, 231.2879181, -95.2828522, 183.4892120, -303.7785645, 326.5707397

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4060023, upper bound: 339.4203258
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4060023, upper bound: 339.4207969
time: 1.11 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -141.8469238, 213.3953247, -127.7809448, 193.0737152, -334.9206238, 341.1762695
1: -112.3477783, 206.1880798, -101.5315018, 186.6283264, -298.9760742, 307.7195740
2: -97.0767288, 211.9947510, -87.6404724, 191.5331116, -288.6098328, 299.6351929
3: -147.1434784, 208.9764252, -132.5283203, 189.1397858, -336.2832642, 341.5047607
4: -116.7677536, 224.4619141, -105.5097198, 202.5829163, -319.3506165, 329.9716187

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4074339, upper bound: 339.4164215
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4059145, upper bound: 339.4157095
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -146.0002441, 220.0361328, -128.2220154, 193.7435608, -339.7437744, 348.2581482
1: -115.6960220, 212.2665558, -101.8790588, 187.2569122, -302.9529419, 314.1455994
2: -100.0173492, 218.4105225, -87.9419327, 192.1886902, -292.2060547, 306.3524475
3: -151.6720581, 215.1582336, -132.9871063, 189.7785339, -341.4505920, 348.1453247
4: -120.2893677, 231.2879181, -105.8677216, 203.2804871, -323.5698242, 337.1556396

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4164080, upper bound: 339.4263045
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4166342, upper bound: 339.4211304
time: 1.21 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 4.39 seconds
NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4135961, upper bound: 339.4139671
NS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4135961, upper bound: 339.4140459
NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4062717, upper bound: 339.4233586
NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4062717, upper bound: 339.4236891
NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4139285, upper bound: 339.4135774
NS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4139286, upper bound: 339.4135774
NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4050551, upper bound: 339.4232740
NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4139286, upper bound: 339.4233141
NS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4135774, upper bound: 339.4208297
NS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4135774, upper bound: 339.4209324
NS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4135774, upper bound: 339.4249188
NS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4135774, upper bound: 339.4262130
NS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4168293, upper bound: 339.4211158
NS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4168293, upper bound: 339.4211158
NS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4034707, upper bound: 339.4251675
NS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4168293, upper bound: 339.4259880
NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4209257, upper bound: 339.4196022
NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4209257, upper bound: 339.4196022
NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4209257, upper bound: 339.4196022
NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4209257, upper bound: 339.4196022
NS_A2_B2_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4164824, upper bound: 339.4218053
NS_A2_B2_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4164824, upper bound: 339.4218053
NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4165926, upper bound: 339.4230750
NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4165926, upper bound: 339.4230750
NS_A2_B2_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4169142, upper bound: 339.4218250
NS_A2_B2_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4169142, upper bound: 339.4218250
NS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4169142, upper bound: 339.4227968
NS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4169142, upper bound: 339.4227968
NS_A2_B2_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4062056, upper bound: 339.4197208
NS_A2_B2_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4062056, upper bound: 339.4197208
NS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4062056, upper bound: 339.4197208
NS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4062056, upper bound: 339.4197208
NS_A2_B2_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4137689, upper bound: 339.4213627
NS_A2_B2_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4137519, upper bound: 339.4137157
NS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4137689, upper bound: 339.4213951
NS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4137583, upper bound: 339.4137524
NS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4138078, upper bound: 339.4202581
NS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4138078, upper bound: 339.4202581
NS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4138078, upper bound: 339.4202581
NS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4138078, upper bound: 339.4202581
NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4060023, upper bound: 339.4203258
NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4060023, upper bound: 339.4207969
NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4074339, upper bound: 339.4164215
NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4059145, upper bound: 339.4157095
NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4164080, upper bound: 339.4263045
NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.39
Output dim: 4, lower bound: -339.4166342, upper bound: 339.4211304

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -119.3333206, 180.3211670, -122.5723801, 185.2850342, -304.6182861, 302.8935547
1: -95.0598526, 174.2033386, -97.7203217, 179.0567474, -274.1165466, 271.9236450
2: -81.9842834, 178.8122864, -84.2710648, 183.7273407, -265.7116089, 263.0833130
3: -124.0454788, 176.7102661, -127.4514313, 181.7063293, -305.7518005, 304.1616821
4: -98.7374420, 189.1328430, -101.4752197, 194.3064423, -293.0438538, 290.6080017

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -119.3333206, 180.3211670, -119.3333206, 180.3211670, -299.6544800, 299.6544800
1: -95.0598526, 174.2033386, -95.0598526, 174.2033386, -269.2631836, 269.2631836
2: -81.9842834, 178.8122864, -81.9842834, 178.8122864, -260.7965698, 260.7965698
3: -124.0454788, 176.7102661, -124.0454788, 176.7102661, -300.7557373, 300.7557373
4: -98.7374420, 189.1328430, -98.7374420, 189.1328430, -287.8702698, 287.8702698

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -119.3333206, 180.3211670, -120.6980896, 182.1997681, -301.5330505, 301.0192566
1: -95.0598526, 174.2033386, -96.0438080, 176.1367798, -271.1965942, 270.2471313
2: -81.9842834, 178.8122864, -82.8602142, 180.7209473, -262.7052307, 261.6724854
3: -124.0454788, 176.7102661, -125.3560944, 178.6244965, -302.6699829, 302.0663452
4: -98.7374420, 189.1328430, -99.7380981, 191.1845703, -289.9219971, 288.8709412

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -119.3333206, 180.3211670, -117.5599213, 177.4074860, -296.7408142, 297.8811035
1: -95.0598526, 174.2033386, -93.4347382, 171.4351501, -266.4949951, 267.6380615
2: -81.9842834, 178.8122864, -80.6119690, 175.9689789, -257.9532471, 259.4242249
3: -124.0454788, 176.7102661, -122.0255203, 173.7874908, -297.8329163, 298.7357788
4: -98.7374420, 189.1328430, -97.0450668, 186.1791840, -284.9166260, 286.1778259

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -120.6980896, 182.1997681, -122.5723801, 185.2850342, -305.9830933, 304.7721252
1: -96.0438080, 176.1367798, -97.7203217, 179.0567474, -275.1005249, 273.8570862
2: -82.8602142, 180.7209473, -84.2710648, 183.7273407, -266.5875549, 264.9920044
3: -125.3560944, 178.6244965, -127.4514313, 181.7063293, -307.0623779, 306.0759277
4: -99.7380981, 191.1845703, -101.4752197, 194.3064423, -294.0445557, 292.6597290

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -120.6980896, 182.1997681, -119.3333206, 180.3211670, -301.0192566, 301.5330505
1: -96.0438080, 176.1367798, -95.0598526, 174.2033386, -270.2471313, 271.1965942
2: -82.8602142, 180.7209473, -81.9842834, 178.8122864, -261.6724854, 262.7052307
3: -125.3560944, 178.6244965, -124.0454788, 176.7102661, -302.0663452, 302.6699829
4: -99.7380981, 191.1845703, -98.7374420, 189.1328430, -288.8709412, 289.9219971

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -117.5599213, 177.4074860, -122.5723801, 185.2850342, -302.8449402, 299.9798584
1: -93.4347382, 171.4351501, -97.7203217, 179.0567474, -272.4914551, 269.1554565
2: -80.6119690, 175.9689789, -84.2710648, 183.7273407, -264.3392944, 260.2400208
3: -122.0255203, 173.7874908, -127.4514313, 181.7063293, -303.7318420, 301.2388916
4: -97.0450668, 186.1791840, -101.4752197, 194.3064423, -291.3514099, 287.6543884

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -117.5599213, 177.4074860, -119.3333206, 180.3211670, -297.8811035, 296.7408142
1: -93.4347382, 171.4351501, -95.0598526, 174.2033386, -267.6380615, 266.4949951
2: -80.6119690, 175.9689789, -81.9842834, 178.8122864, -259.4242249, 257.9532471
3: -122.0255203, 173.7874908, -124.0454788, 176.7102661, -298.7357788, 297.8329163
4: -97.0450668, 186.1791840, -98.7374420, 189.1328430, -286.1778259, 284.9166260

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -120.6980896, 182.1997681, -120.6980896, 182.1997681, -302.8978271, 302.8978271
1: -96.0438080, 176.1367798, -96.0438080, 176.1367798, -272.1805420, 272.1805725
2: -82.8602142, 180.7209473, -82.8602142, 180.7209473, -263.5811768, 263.5811768
3: -125.3560944, 178.6244965, -125.3560944, 178.6244965, -303.9805603, 303.9805603
4: -99.7380981, 191.1845703, -99.7380981, 191.1845703, -290.9226685, 290.9226685

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4140284, upper bound: 339.4148080
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4136607, upper bound: 339.4148106
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -120.6980896, 182.1997681, -117.5599213, 177.4074860, -298.1055908, 299.7596741
1: -96.0438080, 176.1367798, -93.4347382, 171.4351501, -267.4789429, 269.5715027
2: -82.8602142, 180.7209473, -80.6119690, 175.9689789, -258.8291626, 261.3328857
3: -125.3560944, 178.6244965, -122.0255203, 173.7874908, -299.1434631, 300.6500244
4: -99.7380981, 191.1845703, -97.0450668, 186.1791840, -285.9172974, 288.2295837

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4140284, upper bound: 339.4148080
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4136607, upper bound: 339.4148106
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -117.5599213, 177.4074860, -120.6980896, 182.1997681, -299.7596436, 298.1055908
1: -93.4347382, 171.4351501, -96.0438080, 176.1367798, -269.5715027, 267.4789429
2: -80.6119690, 175.9689789, -82.8602142, 180.7209473, -261.3329163, 258.8291931
3: -122.0255203, 173.7874908, -125.3560944, 178.6244965, -300.6500244, 299.1434631
4: -97.0450668, 186.1791840, -99.7380981, 191.1845703, -288.2295837, 285.9172974

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4124735, upper bound: 339.4165982
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4123355, upper bound: 339.4165162
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -117.5599213, 177.4074860, -117.5599213, 177.4074860, -294.9674072, 294.9674072
1: -93.4347382, 171.4351501, -93.4347382, 171.4351501, -264.8698730, 264.8698730
2: -80.6119690, 175.9689789, -80.6119690, 175.9689789, -256.5809326, 256.5809326
3: -122.0255203, 173.7874908, -122.0255203, 173.7874908, -295.8129272, 295.8129272
4: -97.0450668, 186.1791840, -97.0450668, 186.1791840, -283.2242432, 283.2242126

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4124735, upper bound: 339.4165982
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4123355, upper bound: 339.4165162
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -109.4973755, 164.9560242, -125.8211365, 189.4454956, -298.9428406, 290.7771606
1: -87.0287170, 159.2531433, -99.9073944, 182.3699951, -269.3987122, 259.1605225
2: -75.0659561, 163.6335602, -86.1419907, 187.4766083, -262.5425720, 249.7755432
3: -113.8193054, 161.5401459, -130.6053619, 185.2548218, -299.0740967, 292.1454468
4: -90.3632507, 173.2369843, -103.8340988, 198.3768921, -288.7400513, 277.0710754

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -109.4973755, 164.9560242, -133.7337341, 200.4956360, -309.9928894, 298.6897583
1: -87.0287170, 159.2531433, -105.8746490, 193.1281738, -280.1568909, 265.1277771
2: -75.0659561, 163.6335602, -91.3699951, 198.7626190, -273.8284912, 255.0035248
3: -113.8193054, 161.5401459, -138.5630798, 196.1777344, -309.9970398, 300.1031189
4: -90.3632507, 173.2369843, -110.1407776, 210.4121857, -300.7752991, 283.3777466

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -118.5599899, 177.7195282, -125.8211365, 189.4454956, -308.0054626, 303.5406494
1: -93.7300568, 171.4063721, -99.9073944, 182.3699951, -276.1000061, 271.3137207
2: -80.9339523, 176.3727417, -86.1419907, 187.4766083, -268.4105530, 262.5146790
3: -122.6427002, 173.9042206, -130.6053619, 185.2548218, -307.8974915, 304.5095215
4: -97.4791794, 186.7216034, -103.8340988, 198.3768921, -295.8560791, 290.5556946

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -118.5599899, 177.7195282, -133.7337341, 200.4956360, -319.0555115, 311.4532471
1: -93.7300568, 171.4063721, -105.8746490, 193.1281738, -286.8582153, 277.2810059
2: -80.9339523, 176.3727417, -91.3699951, 198.7626190, -279.6965637, 267.7426758
3: -122.6427002, 173.9042206, -138.5630798, 196.1777344, -318.8204346, 312.4671936
4: -97.4791794, 186.7216034, -110.1407776, 210.4121857, -307.8913269, 296.8623657

Time for backsubstitution: 1.60 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.66 + 417.81 = 421.47 seconds
