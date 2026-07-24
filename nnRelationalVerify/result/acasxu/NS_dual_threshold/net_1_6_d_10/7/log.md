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
execution time: IAR + RelationalAnalysis = 0.73 + 2.54 = 3.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -86.5211207, upper bound: 86.5211207

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5182425, upper bound: 86.5161715
time: 0.59 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5211127, upper bound: 86.5211127
time: 0.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.38 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 0, lower bound: -86.5182425, upper bound: 86.5161715
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 0, lower bound: -86.5211127, upper bound: 86.5211127

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -31.8156433, 70.5592194, -31.1796970, 68.6902695, -100.5059052, 101.7389145
1: -66.6389771, 105.4871902, -65.2583008, 102.6108856, -169.2498627, 170.7454681
2: -50.6282349, 103.3955612, -49.5800209, 100.8177261, -151.4459534, 152.9755554
3: -76.9255829, 123.2303696, -75.3810501, 119.9454651, -196.8710480, 198.6114197
4: -70.2429886, 117.3958817, -68.8452377, 114.2890854, -184.5320740, 186.2410889

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_B1

### Relational analysis result of NS_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5165012, upper bound: 86.5154523
time: 0.61 seconds

## Relational analysis of NS_B1_B2

### Relational analysis result of NS_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5164189, upper bound: 86.5152749
time: 0.68 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -31.8210735, 70.5720062, -31.8100567, 70.5460663, -102.3671341, 102.3820648
1: -66.6505203, 105.5066299, -66.6271591, 105.4671860, -172.1176758, 172.1337891
2: -50.6369743, 103.4140015, -50.6192589, 103.3764648, -154.0134277, 154.0332489
3: -76.9388275, 123.2527008, -76.9119568, 123.2070084, -200.1458435, 200.1646576
4: -70.2549973, 117.4171600, -70.2305679, 117.3736649, -187.6286621, 187.6477203

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_B1

### Relational analysis result of NS_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5174207, upper bound: 86.5185661
time: 0.63 seconds

## Relational analysis of NS_B2_B2

### Relational analysis result of NS_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5164189, upper bound: 86.5169677
time: 0.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.22 seconds
NS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -86.5165012, upper bound: 86.5154523
NS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -86.5164189, upper bound: 86.5152749
NS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -86.5174207, upper bound: 86.5185661
NS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -86.5164189, upper bound: 86.5169677

## BFS NS instance: NS_B1_B1

### Backsubstitution after applying NS history:
0: -31.5209026, 69.8698807, -28.5541763, 62.5513191, -94.0722198, 98.4240494
1: -65.9349747, 104.4508438, -59.0228004, 93.3707809, -159.3057098, 163.4736481
2: -50.1371346, 102.2901230, -45.2098618, 91.0363464, -141.1734772, 147.4999847
3: -76.1779633, 121.9272079, -68.7466431, 108.3766937, -184.5546417, 190.6738586
4: -69.6057587, 116.1853638, -63.1495056, 103.5553741, -173.1611176, 179.3348236

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A1

### Relational analysis result of NS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5164189, upper bound: 86.5152749
time: 0.72 seconds

## Relational analysis of NS_B1_B1_A2

### Relational analysis result of NS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5164189, upper bound: 86.5152749
time: 0.65 seconds

## BFS NS instance: NS_B1_B2

### Backsubstitution after applying NS history:
0: -30.5649128, 67.8282089, -38.4424171, 83.5763550, -114.1412659, 106.2706146
1: -63.9659348, 101.3031387, -78.4060593, 124.4601669, -188.4260864, 179.7091827
2: -48.6359596, 99.4419937, -60.3226204, 121.4542847, -170.0902405, 159.7646179
3: -73.9201355, 118.5430145, -91.7294617, 144.7534790, -218.6736145, 210.2724609
4: -67.4989929, 112.7740097, -84.7248688, 137.9193878, -205.4183807, 197.4988708

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B2_A1

### Relational analysis result of NS_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5164189, upper bound: 86.5152749
time: 0.66 seconds

## Relational analysis of NS_B1_B2_A2

### Relational analysis result of NS_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5164189, upper bound: 86.5152749
time: 0.61 seconds

## BFS NS instance: NS_B2_B1

### Backsubstitution after applying NS history:
0: -31.5263271, 69.8826981, -29.1720409, 64.3802185, -95.9065475, 99.0547409
1: -65.9465027, 104.4702988, -60.3512383, 96.1942291, -162.1407318, 164.8215027
2: -50.1458817, 102.3085556, -46.2266121, 93.5646439, -143.7105255, 148.5351715
3: -76.1912155, 121.9495392, -70.2400208, 111.6104431, -187.8016052, 192.1895599
4: -69.6177521, 116.2066345, -64.5171280, 106.5881348, -176.2058868, 180.7237549

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_B1_A1

### Relational analysis result of NS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169677, upper bound: 86.5169677
time: 0.66 seconds

## Relational analysis of NS_B2_B1_A2

### Relational analysis result of NS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169677, upper bound: 86.5169677
time: 0.69 seconds

## BFS NS instance: NS_B2_B2

### Backsubstitution after applying NS history:
0: -30.5702000, 67.8408203, -42.9827538, 94.9132767, -125.4834595, 110.8209152
1: -63.9771767, 101.3224030, -87.6415253, 141.6354065, -205.6125641, 188.9638977
2: -48.6444740, 99.4600906, -67.5333633, 136.7691498, -185.4136200, 166.9934082
3: -73.9330521, 118.5650711, -102.6624832, 163.6735077, -237.6065674, 221.2275238
4: -67.5106735, 112.7951813, -94.6632690, 156.5920410, -224.1027069, 207.4584503

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_B2_A1

### Relational analysis result of NS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169677, upper bound: 86.5169677
time: 0.69 seconds

## Relational analysis of NS_B2_B2_A2

### Relational analysis result of NS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169677, upper bound: 86.5169677
time: 0.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.16 seconds
NS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.16
Output dim: 0, lower bound: -86.5164189, upper bound: 86.5152749
NS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.16
Output dim: 0, lower bound: -86.5164189, upper bound: 86.5152749
NS_B1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.16
Output dim: 0, lower bound: -86.5164189, upper bound: 86.5152749
NS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.16
Output dim: 0, lower bound: -86.5164189, upper bound: 86.5152749
NS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.16
Output dim: 0, lower bound: -86.5169677, upper bound: 86.5169677
NS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.16
Output dim: 0, lower bound: -86.5169677, upper bound: 86.5169677
NS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.16
Output dim: 0, lower bound: -86.5169677, upper bound: 86.5169677
NS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.16
Output dim: 0, lower bound: -86.5169677, upper bound: 86.5169677

## BFS NS instance: NS_B1_B1_A1

### Backsubstitution after applying NS history:
0: -29.1775131, 64.3929214, -28.5541763, 62.5513191, -91.7288361, 92.9470978
1: -60.3626328, 96.2134781, -59.0228004, 93.3707809, -153.7333984, 155.2362823
2: -46.2353096, 93.5830002, -45.2098618, 91.0363464, -137.2716217, 138.7928619
3: -70.2531738, 111.6330338, -68.7466431, 108.3766937, -178.6298676, 180.3796692
4: -64.5292282, 106.6093903, -63.1495056, 103.5553741, -168.0845947, 169.7588806

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A1_A1

### Relational analysis result of NS_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5164877, upper bound: 86.5154523
time: 0.65 seconds

## Relational analysis of NS_B1_B1_A1_A2

### Relational analysis result of NS_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5163414, upper bound: 86.5154523
time: 0.88 seconds

## BFS NS instance: NS_B1_B1_A2

### Backsubstitution after applying NS history:
0: -42.5144348, 93.7747498, -28.5541763, 62.5513191, -104.9949036, 122.3289185
1: -86.6941605, 139.8611298, -59.0228004, 93.3707809, -180.0649261, 198.8839264
2: -66.7966156, 135.1631622, -45.2098618, 91.0363464, -157.8329620, 180.3729858
3: -101.5294342, 161.6704407, -68.7466431, 108.3766937, -209.9061279, 230.4170837
4: -93.6414337, 154.7103119, -63.1495056, 103.5553741, -196.9038239, 217.8598175

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A2_B1

### Relational analysis result of NS_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5160591, upper bound: 86.5149549
time: 0.60 seconds

## Relational analysis of NS_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A2_A1

### Relational analysis result of NS_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148376, upper bound: 86.5149347
time: 0.58 seconds

## Relational analysis of NS_B1_B1_A2_A2

### Relational analysis result of NS_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148376, upper bound: 86.5154523
time: 0.60 seconds

## BFS NS instance: NS_B1_B2_A1

### Backsubstitution after applying NS history:
0: -29.1775131, 64.3929214, -38.4345551, 83.5572586, -112.7347717, 102.8274765
1: -60.3626328, 96.2134781, -78.3872299, 124.4304733, -184.7930908, 174.6007080
2: -46.2353096, 93.5830002, -60.3096390, 121.4256821, -167.6609650, 153.8926392
3: -70.2531738, 111.6330338, -91.7095108, 144.7196808, -214.9728546, 203.3425293
4: -64.5292282, 106.6093903, -84.7076721, 137.8866425, -202.4158478, 191.3170471

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B2_A1_A1

### Relational analysis result of NS_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5164052, upper bound: 86.5152749
time: 1.50 seconds

## Relational analysis of NS_B1_B2_A1_A2

### Relational analysis result of NS_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5162587, upper bound: 86.5152749
time: 0.57 seconds

## BFS NS instance: NS_B1_B2_A2

### Backsubstitution after applying NS history:
0: -43.0562057, 95.0901031, -38.4424171, 83.5763550, -126.3670425, 133.1436615
1: -87.7890930, 141.9098816, -78.4060593, 124.4601669, -212.2492523, 220.3159485
2: -67.6488571, 137.0177765, -60.3226204, 121.4542847, -189.1031494, 197.3403931
3: -102.8393021, 163.9830017, -91.7294617, 144.7534790, -247.5927734, 255.7124634
4: -94.8237381, 156.8841248, -84.7248688, 137.9193878, -232.3104858, 241.0080566

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B2_A2_A1

### Relational analysis result of NS_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5164052, upper bound: 86.5152749
time: 0.69 seconds

## Relational analysis of NS_B1_B2_A2_A2

### Relational analysis result of NS_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5162587, upper bound: 86.5152749
time: 0.60 seconds

## BFS NS instance: NS_B2_B1_A1

### Backsubstitution after applying NS history:
0: -29.1829376, 64.4057617, -29.1720409, 64.3802185, -93.5631561, 93.5777969
1: -60.3741684, 96.2330017, -60.3512383, 96.1942291, -156.5683899, 156.5842133
2: -46.2440414, 93.6015091, -46.2266121, 93.5646439, -139.8086548, 139.8280945
3: -70.2664032, 111.6556091, -70.2400208, 111.6104431, -181.8768158, 181.8956299
4: -64.5412216, 106.6308975, -64.5171280, 106.5881348, -171.1293335, 171.1480255

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_B1_A1_B1

### Relational analysis result of NS_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5174207, upper bound: 86.5185661
time: 0.66 seconds

## Relational analysis of NS_B2_B1_A1_B2

### Relational analysis result of NS_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5173880, upper bound: 86.5183279
time: 1.22 seconds

## BFS NS instance: NS_B2_B1_A2

### Backsubstitution after applying NS history:
0: -42.8066521, 94.4861526, -29.1720409, 64.3802185, -107.1868744, 123.6581955
1: -87.2865219, 140.9687958, -60.3512383, 96.1942291, -183.4806671, 201.3200226
2: -67.2566833, 136.1674652, -46.2266121, 93.5646439, -160.8213043, 182.3940735
3: -102.2370224, 162.9212189, -70.2400208, 111.6104431, -213.8474579, 233.1612396
4: -94.2791901, 155.8873138, -64.5171280, 106.5881348, -200.7441711, 220.4044495

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_B1_A2_B1

### Relational analysis result of NS_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5174207, upper bound: 86.5185661
time: 0.69 seconds

## Relational analysis of NS_B2_B1_A2_B2

### Relational analysis result of NS_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5173880, upper bound: 86.5183279
time: 0.65 seconds

## BFS NS instance: NS_B2_B2_A1

### Backsubstitution after applying NS history:
0: -29.1829376, 64.4057617, -42.7576866, 94.3679733, -123.5509109, 107.1634445
1: -60.3741684, 96.2330017, -87.1881485, 140.7864075, -201.1605835, 183.4211273
2: -46.2440414, 93.6015091, -67.1795731, 136.0016022, -182.2456360, 160.7810669
3: -70.2664032, 111.6556091, -102.1193161, 162.7158508, -232.9822540, 213.7749329
4: -64.5412216, 106.6308975, -94.1723709, 155.6917114, -220.2329254, 200.6832733

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_B2_A1_A1

### Relational analysis result of NS_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148236, upper bound: 86.5141301
time: 0.72 seconds

## Relational analysis of NS_B2_B2_A1_A2

### Relational analysis result of NS_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5134377, upper bound: 86.5134377
time: 0.75 seconds

## BFS NS instance: NS_B2_B2_A2

### Backsubstitution after applying NS history:
0: -43.0617256, 95.1030045, -43.0506821, 95.0768280, -137.6446533, 137.6598969
1: -87.8006973, 141.9292755, -87.7773285, 141.8895416, -229.6902161, 229.7065735
2: -67.6576920, 137.0364685, -67.6398010, 136.9982452, -204.6559448, 204.6762695
3: -102.8526154, 164.0055847, -102.8255920, 163.9594269, -266.8119812, 266.8311768
4: -94.8358154, 156.9057007, -94.8113403, 156.8616486, -250.8192596, 250.8393555

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B2_B2_A2_A1

### Relational analysis result of NS_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149630, upper bound: 86.5149696
time: 0.76 seconds

## Relational analysis of NS_B2_B2_A2_A2

### Relational analysis result of NS_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145660, upper bound: 86.5145660
time: 0.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.37 seconds
NS_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -86.5164877, upper bound: 86.5154523
NS_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -86.5163414, upper bound: 86.5154523
NS_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -86.5148376, upper bound: 86.5149347
NS_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -86.5148376, upper bound: 86.5154523
NS_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -86.5164052, upper bound: 86.5152749
NS_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -86.5162587, upper bound: 86.5152749
NS_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -86.5164052, upper bound: 86.5152749
NS_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -86.5162587, upper bound: 86.5152749
NS_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -86.5174207, upper bound: 86.5185661
NS_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -86.5173880, upper bound: 86.5183279
NS_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -86.5174207, upper bound: 86.5185661
NS_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -86.5173880, upper bound: 86.5183279
NS_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -86.5148236, upper bound: 86.5141301
NS_B2_B2_A1_A2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -86.5134377, upper bound: 86.5134377
NS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -86.5149630, upper bound: 86.5149696
NS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -86.5145660, upper bound: 86.5145660

## BFS NS instance: NS_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -28.5678158, 62.9459915, -28.4062786, 62.2019920, -90.7698059, 91.3522720
1: -59.0059357, 93.9977036, -58.6970520, 92.8344116, -151.8403473, 152.6947174
2: -45.2425537, 91.4281693, -44.9698601, 90.5157852, -135.7583313, 136.3979797
3: -68.7455826, 108.9839172, -68.3823853, 107.7285309, -176.4740753, 177.3663025
4: -63.1869850, 104.1404037, -62.8234444, 102.9609146, -166.1479034, 166.9638214

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_B1_A1_A1_B1

### Relational analysis result of NS_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5176361, upper bound: 86.5151504
time: 0.61 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A1_A1_A1

### Relational analysis result of NS_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5181036, upper bound: 86.5158956
time: 0.68 seconds

## Relational analysis of NS_B1_B1_A1_A1_A2

### Relational analysis result of NS_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5179560, upper bound: 86.5157085
time: 0.98 seconds

## BFS NS instance: NS_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -31.3124104, 68.6801529, -27.9076805, 61.0260773, -92.3384781, 96.5878296
1: -64.6362915, 102.6246567, -57.5935936, 91.0669327, -155.7032166, 160.2182312
2: -49.5336800, 99.8288574, -44.1581345, 88.7856064, -138.3192902, 143.9869690
3: -75.3951416, 119.0113678, -67.1418228, 105.7145233, -181.1096039, 186.1531830
4: -69.2435608, 113.6754608, -61.7215424, 100.9783859, -170.2219543, 175.3969727

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A1_A2_B1

### Relational analysis result of NS_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5161870, upper bound: 86.5151513
time: 0.62 seconds

## Relational analysis of NS_B1_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A1_A2_A1

### Relational analysis result of NS_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5166352, upper bound: 86.5155211
time: 0.65 seconds

## Relational analysis of NS_B1_B1_A1_A2_A2

### Relational analysis result of NS_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5166287, upper bound: 86.5154611
time: 0.61 seconds

## BFS NS instance: NS_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -41.7150764, 91.5614014, -28.5541763, 62.5513191, -104.2064056, 120.1155624
1: -84.9965286, 136.5135651, -59.0228004, 93.3707809, -178.3672943, 195.5363464
2: -65.4855270, 132.1215210, -45.2098618, 91.0363464, -156.5218811, 177.3313751
3: -99.5876694, 157.8203888, -68.7466431, 108.3766937, -207.9643555, 226.5670319
4: -91.8982162, 151.0429230, -63.1495056, 103.5553741, -195.2222290, 214.1924286

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

## BFS NS instance: NS_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -42.5083275, 93.7606354, -28.5541763, 62.5513191, -104.9890594, 122.3147964
1: -86.6816406, 139.8395538, -59.0228004, 93.3707809, -180.0524292, 198.8623352
2: -66.7869720, 135.1423798, -45.2098618, 91.0363464, -157.8233032, 180.3522339
3: -101.5148392, 161.6453400, -68.7466431, 108.3766937, -209.8915253, 230.3919830
4: -93.6282349, 154.6864777, -63.1495056, 103.5553741, -196.8908081, 217.8359680

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

## BFS NS instance: NS_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -28.5678158, 62.9459915, -38.2421379, 83.1033096, -111.6711273, 101.1881256
1: -59.0059357, 93.9977036, -77.9470291, 123.7415009, -182.7474365, 171.9447021
2: -45.2425537, 91.4281693, -59.9924965, 120.7244568, -165.9670105, 151.4206390
3: -68.7455826, 108.9839172, -91.2260513, 143.8768768, -212.6224365, 200.2099457
4: -63.1869850, 104.1404037, -84.2845001, 137.1091309, -200.2960968, 188.4248810

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_B2_A1_A1_B1

### Relational analysis result of NS_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5174125, upper bound: 86.5147311
time: 0.62 seconds

## Relational analysis of NS_B1_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B2_A1_A1_A1

### Relational analysis result of NS_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151148, upper bound: 86.5127515
time: 0.72 seconds

## Relational analysis of NS_B1_B2_A1_A1_A2

### Relational analysis result of NS_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5179964, upper bound: 86.5158364
time: 0.64 seconds

## BFS NS instance: NS_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -31.3124104, 68.6801529, -37.8740997, 82.2315445, -113.5439529, 106.5542526
1: -64.6362915, 102.6246567, -77.1337280, 122.4176407, -187.0539246, 179.7583618
2: -49.5336800, 99.8288574, -59.3919373, 119.4658127, -168.9994965, 159.2207947
3: -75.3951416, 119.0113678, -90.3073196, 142.3956451, -217.7907562, 209.3186951
4: -69.2435608, 113.6754608, -83.4702606, 135.6286011, -204.8721619, 197.1457062

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_B2_A1_A2_A1

### Relational analysis result of NS_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5165472, upper bound: 86.5153383
time: 0.99 seconds

## Relational analysis of NS_B1_B2_A1_A2_A2

### Relational analysis result of NS_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5165367, upper bound: 86.5152786
time: 0.60 seconds

## BFS NS instance: NS_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -42.3098297, 93.3649521, -38.2549171, 83.1343384, -125.1636505, 131.1950531
1: -86.1018829, 139.3482513, -77.9775772, 123.7900314, -209.8918457, 217.3258209
2: -66.4185562, 134.4545898, -60.0136337, 120.7709351, -187.1894836, 194.4682312
3: -100.9816742, 160.9483337, -91.2585068, 143.9322357, -244.9139099, 252.2068481
4: -93.1812286, 153.9613953, -84.3125153, 137.1622620, -229.8710632, 237.5846710

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_B2_A2_A1_A1

### Relational analysis result of NS_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5160544, upper bound: 86.5151141
time: 0.66 seconds

## Relational analysis of NS_B1_B2_A2_A1_A2

### Relational analysis result of NS_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5163167, upper bound: 86.5151683
time: 0.97 seconds

## BFS NS instance: NS_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -45.5305443, 99.9822693, -37.8761902, 82.2366333, -127.4474182, 137.4532623
1: -92.6673660, 149.2157898, -77.1387405, 122.4256134, -215.0929871, 226.3545074
2: -71.4414597, 144.1277771, -59.3954010, 119.4734497, -190.9149017, 203.5231781
3: -108.7424469, 172.3611908, -90.3126297, 142.4047241, -251.1225586, 262.6738281
4: -100.3060226, 164.9200592, -83.4748611, 135.6372833, -235.3283386, 247.7680054

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_B2_A2_A2_A1

### Relational analysis result of NS_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5158230, upper bound: 86.5150823
time: 0.69 seconds

## Relational analysis of NS_B1_B2_A2_A2_A2

### Relational analysis result of NS_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5161787, upper bound: 86.5151683
time: 0.67 seconds

## BFS NS instance: NS_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -27.9773865, 61.7414780, -26.3395023, 58.1513901, -86.1287689, 88.0809784
1: -57.6542511, 92.2282104, -53.9966736, 86.8734818, -144.5277252, 146.2248840
2: -44.2660332, 89.4835358, -41.5833473, 83.9544678, -128.2205048, 131.0668640
3: -67.2873764, 106.7991257, -63.2399521, 100.3388596, -167.6262360, 170.0390778
4: -61.9157143, 102.0399323, -58.3398476, 95.8914261, -157.8070984, 160.3797760

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B1_A1_B1_A1

### Relational analysis result of NS_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5188107, upper bound: 86.5188107
time: 0.76 seconds

## Relational analysis of NS_B2_B1_A1_B1_A2

### Relational analysis result of NS_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5188107, upper bound: 86.5188107
time: 0.69 seconds

## BFS NS instance: NS_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -28.2654629, 62.4260864, -35.3117943, 78.0757370, -106.3412018, 97.7378845
1: -58.4345589, 93.2649155, -71.8210373, 116.2003860, -174.6349487, 165.0858917
2: -44.7808723, 90.6591492, -55.4195900, 112.1926270, -156.9734955, 146.0787354
3: -68.0360336, 108.1961823, -84.4210052, 134.1277313, -202.1637421, 192.6171570
4: -62.5292778, 103.3136292, -77.9112930, 128.2769928, -190.8062592, 181.2249146

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B1_A1_B2_A1

### Relational analysis result of NS_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5188107, upper bound: 86.5188107
time: 1.27 seconds

## Relational analysis of NS_B2_B1_A1_B2_A2

### Relational analysis result of NS_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5188107, upper bound: 86.5188107
time: 1.14 seconds

## BFS NS instance: NS_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -41.4873276, 91.5164642, -26.3395023, 58.1513901, -99.6317902, 117.8559647
1: -84.2661972, 136.5941315, -53.9966736, 86.8734818, -171.1396484, 190.5908051
2: -65.0668869, 131.6334534, -41.5833473, 83.9544678, -149.0213470, 173.2167969
3: -98.9523010, 157.6431274, -63.2399521, 100.3388596, -199.2911377, 220.8830719
4: -91.3962708, 150.8023834, -58.3398476, 95.8914261, -187.1130981, 209.1422272

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_B1_A1

### Relational analysis result of NS_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5168967, upper bound: 86.5181542
time: 1.30 seconds

## Relational analysis of NS_B2_B1_A2_B1_A2

### Relational analysis result of NS_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5167379, upper bound: 86.5181104
time: 0.69 seconds

## BFS NS instance: NS_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -42.1042442, 92.9855804, -35.3117943, 78.0757370, -119.7276230, 127.9948425
1: -85.7678223, 138.7189789, -71.8210373, 116.2003860, -201.9682007, 210.5399628
2: -66.1196518, 133.9389343, -55.4195900, 112.1926270, -178.3122864, 189.3585205
3: -100.5047379, 160.3120117, -84.4210052, 134.1277313, -234.6324768, 244.7330017
4: -92.7330856, 153.3569794, -77.9112930, 128.2769928, -220.2526093, 230.8667755

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_B2_A1

### Relational analysis result of NS_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5168583, upper bound: 86.5179077
time: 0.65 seconds

## Relational analysis of NS_B2_B1_A2_B2_A2

### Relational analysis result of NS_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5167104, upper bound: 86.5178760
time: 0.79 seconds

## BFS NS instance: NS_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -28.7743149, 63.4770966, -42.7231293, 94.2895584, -123.0638733, 106.2002258
1: -59.4856415, 94.8242722, -87.1122665, 140.6685638, -200.1541748, 181.9365387
2: -45.5844383, 92.2265701, -67.1235657, 135.8852997, -181.4697418, 159.3501282
3: -69.2623062, 110.0045013, -102.0340958, 162.5776672, -231.8399506, 212.0385895
4: -63.6406403, 105.0588760, -94.0963287, 155.5591888, -219.1998291, 199.0269165

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_B2_A1_A1_B1

### Relational analysis result of NS_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149421, upper bound: 86.5140552
time: 0.72 seconds

## Relational analysis of NS_B2_B2_A1_A1_B2

### Relational analysis result of NS_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149421, upper bound: 86.5140552
time: 0.68 seconds

## BFS NS instance: NS_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -43.1081581, 95.7331009, -42.2956467, 93.5025177, -136.1005707, 137.4466858
1: -87.9880371, 143.1307678, -86.2269516, 139.5245514, -227.5125275, 229.3230133
2: -67.7785492, 137.9443359, -66.4570389, 134.6767120, -202.4552612, 204.4013672
3: -103.0230560, 165.2603912, -101.0053024, 161.2024384, -264.2254944, 266.0932617
4: -94.9564819, 158.0632629, -93.1505280, 154.2556763, -248.2820129, 250.1219788

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_B2_A2_A1_B1

### Relational analysis result of NS_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145660, upper bound: 86.5145660
time: 0.73 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2

### Relational analysis result of NS_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145660, upper bound: 86.5145660
time: 0.77 seconds

## BFS NS instance: NS_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -42.5835991, 94.0922241, -42.8938599, 94.7437210, -136.8319550, 136.4783020
1: -86.7759247, 140.4262085, -87.4402313, 141.3942261, -228.1701508, 227.8664398
2: -66.8843231, 135.5517731, -67.3860474, 136.5076294, -203.3919525, 202.9378204
3: -101.6921005, 162.2522888, -102.4445496, 163.3808594, -265.0729675, 264.6968384
4: -93.7818069, 155.2175446, -94.4656143, 156.3034973, -249.2066040, 248.7736511

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_B2_A2_A2_B1

### Relational analysis result of NS_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145660, upper bound: 86.5145660
time: 0.67 seconds

## Relational analysis of NS_B2_B2_A2_A2_B2

### Relational analysis result of NS_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145660, upper bound: 86.5145660
time: 0.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.11 seconds
NS_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5181036, upper bound: 86.5158956
NS_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5179560, upper bound: 86.5157085
NS_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5166352, upper bound: 86.5155211
NS_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5166287, upper bound: 86.5154611
NS_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5151148, upper bound: 86.5127515
NS_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5179964, upper bound: 86.5158364
NS_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5165472, upper bound: 86.5153383
NS_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5165367, upper bound: 86.5152786
NS_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5160544, upper bound: 86.5151141
NS_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5163167, upper bound: 86.5151683
NS_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5158230, upper bound: 86.5150823
NS_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5161787, upper bound: 86.5151683
NS_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5188107, upper bound: 86.5188107
NS_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5188107, upper bound: 86.5188107
NS_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5188107, upper bound: 86.5188107
NS_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5188107, upper bound: 86.5188107
NS_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5168967, upper bound: 86.5181542
NS_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5167379, upper bound: 86.5181104
NS_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5168583, upper bound: 86.5179077
NS_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5167104, upper bound: 86.5178760
NS_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5149421, upper bound: 86.5140552
NS_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5149421, upper bound: 86.5140552
NS_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5145660, upper bound: 86.5145660
NS_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5145660, upper bound: 86.5145660
NS_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5145660, upper bound: 86.5145660
NS_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -86.5145660, upper bound: 86.5145660

## BFS NS instance: NS_B1_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -28.3568707, 62.4787064, -28.4062786, 62.2019920, -90.5588608, 90.8849792
1: -58.5417290, 93.2915192, -58.6970520, 92.8344116, -151.3761444, 151.9885712
2: -44.8990517, 90.7250900, -44.9698601, 90.5157852, -135.4148254, 135.6949463
3: -68.2270660, 108.1482468, -68.3823853, 107.7285309, -175.9555969, 176.5306396
4: -62.7210770, 103.3423538, -62.8234444, 102.9609146, -165.6819916, 166.1657867

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -28.6560364, 63.1842117, -28.1390400, 61.6146317, -90.2706680, 91.3232498
1: -59.1870308, 94.3849335, -58.1208038, 91.9558792, -151.1428375, 152.5057220
2: -45.3874283, 91.8011551, -44.5413399, 89.6523514, -135.0397491, 136.3424835
3: -68.9751129, 109.4345016, -67.7272568, 106.7050323, -175.6800842, 177.1617584
4: -63.3874626, 104.5763550, -62.2368088, 101.9837189, -165.3711548, 166.8131561

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -31.1023045, 68.2054672, -27.9076805, 61.0260773, -92.1283798, 96.1131439
1: -64.1639633, 101.9123764, -57.5935936, 91.0669327, -155.2308655, 159.5059509
2: -49.1890373, 99.1143646, -44.1581345, 88.7856064, -137.9746399, 143.2724609
3: -74.8737564, 118.1611176, -67.1418228, 105.7145233, -180.5881958, 185.3029480
4: -68.7797165, 112.8693008, -61.7215424, 100.9783859, -169.7581024, 174.5908051

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -31.4743004, 69.1002884, -27.6562996, 60.4842339, -91.9585266, 96.7565842
1: -65.0183182, 103.2694778, -57.0627022, 90.2550125, -155.2733307, 160.3321838
2: -49.8073349, 100.4712753, -43.7578621, 87.9879913, -137.7953186, 144.2291260
3: -75.8275681, 119.7819672, -66.5309448, 104.7701645, -180.5977020, 186.3128967
4: -69.6033401, 114.4127502, -61.1693115, 100.0778275, -169.6811676, 175.5820618

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B1_A1_A2_A2_A1

### Relational analysis result of NS_B1_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149016, upper bound: 86.5149016
time: 0.95 seconds

## Relational analysis of NS_B1_B1_A1_A2_A2_A2

### Relational analysis result of NS_B1_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149016, upper bound: 86.5154611
time: 0.67 seconds

## BFS NS instance: NS_B1_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -25.4616051, 55.8671722, -37.4437675, 81.1608047, -106.6224060, 93.3109436
1: -52.2404213, 83.3287201, -76.1428375, 120.8235931, -173.0640106, 159.4715576
2: -40.2033348, 81.0813217, -58.6778412, 117.8605423, -158.0638733, 139.7591553
3: -61.0892372, 96.5389404, -89.2227707, 140.4132996, -201.5025330, 185.7617188
4: -56.3068504, 92.3237305, -82.5190964, 133.8488007, -190.1556549, 174.8428345

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_B2_A1_A1_A1_A1

### Relational analysis result of NS_B1_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148390, upper bound: 86.5125485
time: 0.61 seconds

## Relational analysis of NS_B1_B2_A1_A1_A1_A2

### Relational analysis result of NS_B1_B2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5149776, upper bound: 86.5118547
time: 0.76 seconds

## BFS NS instance: NS_B1_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -28.3313675, 62.3906288, -38.1485252, 82.8840332, -111.2154007, 100.5391388
1: -58.4675255, 93.1417313, -77.7327652, 123.4079514, -181.8754730, 170.8744659
2: -44.8509598, 90.5909805, -59.8376007, 120.3922958, -165.2432556, 150.4285889
3: -68.1501007, 107.9779282, -90.9896927, 143.4793243, -211.6294250, 198.9675903
4: -62.6634903, 103.1822128, -84.0778809, 136.7317810, -199.3952637, 187.2601013

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_B2_A1_A1_A2_A1

### Relational analysis result of NS_B1_B2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5178436, upper bound: 86.5156933
time: 0.65 seconds

## Relational analysis of NS_B1_B2_A1_A1_A2_A2

### Relational analysis result of NS_B1_B2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5178201, upper bound: 86.5155240
time: 0.76 seconds

## BFS NS instance: NS_B1_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -31.1023045, 68.2054672, -37.8732491, 82.2294922, -113.3317947, 106.0787125
1: -64.1639633, 101.9123764, -77.1317062, 122.4144363, -186.5783691, 179.0440674
2: -49.1890373, 99.1143646, -59.3905411, 119.4627609, -168.6517944, 158.5048981
3: -74.8737564, 118.1611176, -90.3051682, 142.3919983, -217.2657166, 208.4662781
4: -68.7797165, 112.8693008, -83.4684143, 135.6251068, -204.4048157, 196.3376617

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -31.4743004, 69.1002884, -37.5845604, 81.6109161, -113.0852051, 106.6848373
1: -65.0183182, 103.2694778, -76.5249786, 121.4846649, -186.5029907, 179.7944489
2: -49.8073349, 100.4712753, -58.9313622, 118.5472946, -168.3546143, 159.4026337
3: -75.8275681, 119.7819672, -89.6054001, 141.3057098, -217.1332397, 209.3873291
4: -69.6033401, 114.4127502, -82.8319550, 134.5943146, -204.1976624, 197.2447052

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B2_A1_A2_A2_A1

### Relational analysis result of NS_B1_B2_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148109, upper bound: 86.5147187
time: 0.72 seconds

## Relational analysis of NS_B1_B2_A1_A2_A2_A2

### Relational analysis result of NS_B1_B2_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5148109, upper bound: 86.5152786
time: 0.75 seconds

## BFS NS instance: NS_B1_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -42.1146393, 92.9207535, -38.2549171, 83.1343384, -124.9630508, 130.7469177
1: -85.6559601, 138.6894379, -77.9775772, 123.7900314, -209.4459686, 216.6670227
2: -66.0947647, 133.7949219, -60.0136337, 120.7709351, -186.8656921, 193.8085480
3: -100.4946442, 160.1672211, -91.2585068, 143.9322357, -244.4268646, 251.4257202
4: -92.7509079, 153.2077179, -84.3125153, 137.1622620, -229.4266357, 236.8227844

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -42.0789337, 92.9240570, -37.9622498, 82.5064011, -124.3253098, 130.4714203
1: -85.6513290, 138.7010498, -77.3621826, 122.8459702, -208.4972992, 216.0632324
2: -66.0650635, 133.8197632, -59.5481071, 119.8418503, -185.9069214, 193.3678741
3: -100.4550095, 160.1900482, -90.5491867, 142.8300476, -243.2850647, 250.7392120
4: -92.6695862, 153.2612762, -83.6675339, 136.1157990, -228.3659515, 236.2619019

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_B2_A2_A1_A2_A1

### Relational analysis result of NS_B1_B2_A2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5130848, upper bound: 86.5116717
time: 0.68 seconds

## Relational analysis of NS_B1_B2_A2_A1_A2_A2

### Relational analysis result of NS_B1_B2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5162302, upper bound: 86.5151061
time: 0.78 seconds

## BFS NS instance: NS_B1_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -45.3288956, 99.5158081, -37.8761902, 82.2366333, -127.2405014, 136.9821930
1: -92.2025604, 148.5244446, -77.1387405, 122.4256134, -214.6281738, 225.6631470
2: -71.1061478, 143.4319305, -59.3954010, 119.4734497, -190.5795898, 202.8273315
3: -108.2370529, 171.5384369, -90.3126297, 142.4047241, -250.6057587, 261.8510742
4: -99.8628387, 164.1234283, -83.4748611, 135.6372833, -234.8712769, 246.9603119

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_B2_A2_A2_A1_A1

### Relational analysis result of NS_B1_B2_A2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5094315, upper bound: 86.5098741
time: 0.79 seconds

## Relational analysis of NS_B1_B2_A2_A2_A1_A2

### Relational analysis result of NS_B1_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157376, upper bound: 86.5150166
time: 0.67 seconds

## BFS NS instance: NS_B1_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -45.4230385, 99.8294067, -37.5845604, 81.6109161, -126.7359085, 137.0223389
1: -92.4957657, 149.0067139, -76.5249786, 121.4846649, -213.9804077, 225.5316467
2: -71.2913055, 143.9202728, -58.9313622, 118.5472946, -189.8385925, 202.8516388
3: -108.5294876, 172.1185150, -89.6054001, 141.3057098, -249.8352051, 261.7239075
4: -100.0646210, 164.7119598, -82.8319550, 134.5943146, -234.1000061, 246.9430237

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_B2_A2_A2_A2_A1

### Relational analysis result of NS_B1_B2_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5130893, upper bound: 86.5136584
time: 3.23 seconds

## Relational analysis of NS_B1_B2_A2_A2_A2_A2

### Relational analysis result of NS_B1_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5160864, upper bound: 86.5151061
time: 0.81 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -26.3504200, 58.1767349, -26.3395023, 58.1513901, -84.5018082, 84.5162354
1: -54.0193672, 86.9120712, -53.9966736, 86.8734818, -140.8928528, 140.9087524
2: -41.6006546, 83.9912643, -41.5833473, 83.9544678, -125.5551224, 125.5746002
3: -63.2661858, 100.3837051, -63.2399521, 100.3388596, -163.6050415, 163.6236420
4: -58.3637619, 95.9337921, -58.3398476, 95.8914261, -154.2551880, 154.2736053

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_B1_A1_B1

### Relational analysis result of NS_B2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5184728, upper bound: 86.5191232
time: 0.68 seconds

## Relational analysis of NS_B2_B1_A1_B1_A1_B2

### Relational analysis result of NS_B2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5184603, upper bound: 86.5192452
time: 0.73 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -35.3220367, 78.0993195, -26.3395023, 58.1513901, -93.4734192, 104.4388199
1: -71.8421631, 116.2360764, -53.9966736, 86.8734818, -158.7156372, 170.2327423
2: -55.4358253, 112.2268219, -41.5833473, 83.9544678, -139.3902740, 153.8101501
3: -84.4455414, 134.1691895, -63.2399521, 100.3388596, -184.7843781, 197.4091492
4: -77.9337006, 128.3163605, -58.3398476, 95.8914261, -173.8250885, 186.6562042

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_B1_A2_B1

### Relational analysis result of NS_B2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5184728, upper bound: 86.5191232
time: 0.75 seconds

## Relational analysis of NS_B2_B1_A1_B1_A2_B2

### Relational analysis result of NS_B2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5184603, upper bound: 86.5192452
time: 1.03 seconds

## BFS NS instance: NS_B2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -26.3504200, 58.1767349, -35.3117943, 78.0757370, -104.4261551, 93.4885254
1: -54.0193672, 86.9120712, -71.8210373, 116.2003860, -170.2197571, 158.7330475
2: -41.6006546, 83.9912643, -55.4195900, 112.1926270, -153.7932739, 139.4108582
3: -63.2661858, 100.3837051, -84.4210052, 134.1277313, -197.3939209, 184.8046722
4: -58.3637619, 95.9337921, -77.9112930, 128.2769928, -186.6407471, 173.8450928

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_B2_A1_A1

### Relational analysis result of NS_B2_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5182644, upper bound: 86.5183953
time: 0.86 seconds

## Relational analysis of NS_B2_B1_A1_B2_A1_A2

### Relational analysis result of NS_B2_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5183700, upper bound: 86.5183700
time: 1.07 seconds

## BFS NS instance: NS_B2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -35.3220367, 78.0993195, -35.3117943, 78.0757370, -113.1186981, 113.1322327
1: -71.8421631, 116.2360764, -71.8210373, 116.2003860, -188.0425415, 188.0570526
2: -55.4358253, 112.2268219, -55.4195900, 112.1926270, -167.6284485, 167.6464081
3: -84.4455414, 134.1691895, -84.4210052, 134.1277313, -218.5732574, 218.5901794
4: -77.9337006, 128.3163605, -77.9112930, 128.2769928, -205.8903046, 205.9075470

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_B2_A2_B1

### Relational analysis result of NS_B2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5183953, upper bound: 86.5182644
time: 0.73 seconds

## Relational analysis of NS_B2_B1_A1_B2_A2_B2

### Relational analysis result of NS_B2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5183700, upper bound: 86.5183700
time: 0.70 seconds

## BFS NS instance: NS_B2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -39.8138008, 87.9173508, -25.8887253, 57.2000694, -97.0023727, 113.8060684
1: -80.7920303, 131.1452026, -53.0683403, 85.4400482, -166.2320862, 184.2135468
2: -62.4061432, 126.3052521, -40.8717422, 82.5401840, -144.9463196, 167.1769714
3: -94.9102325, 151.3094330, -62.1522636, 98.6653137, -193.5755310, 213.4616699
4: -87.6999283, 144.7586823, -57.3441734, 94.3032074, -181.8129272, 202.1028442

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A2_B1_A1_A1

### Relational analysis result of NS_B2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5156877, upper bound: 86.5175476
time: 0.72 seconds

## Relational analysis of NS_B2_B1_A2_B1_A1_A2

### Relational analysis result of NS_B2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157195, upper bound: 86.5164306
time: 0.65 seconds

## BFS NS instance: NS_B2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -41.7534637, 92.6321945, -25.8492756, 57.0913239, -98.8373032, 118.4814682
1: -84.9656296, 138.3182983, -52.9571228, 85.2710648, -170.2366943, 191.2754211
2: -65.5363998, 133.1421356, -40.7998085, 82.3780899, -147.9144897, 173.9419403
3: -99.6750870, 159.6469574, -62.0451622, 98.4721603, -198.1472473, 221.6921234
4: -91.9901657, 152.7501831, -57.2596550, 94.1184311, -185.9383392, 210.0098267

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B2_B1_A2_B1_A2_A1

### Relational analysis result of NS_B2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5150626, upper bound: 86.5164362
time: 0.99 seconds

## Relational analysis of NS_B2_B1_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_B1_A2_B1

### Relational analysis result of NS_B2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5167379, upper bound: 86.5180506
time: 0.71 seconds

## Relational analysis of NS_B2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A2_B1_A2_A1

### Relational analysis result of NS_B2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157088, upper bound: 86.5175474
time: 0.68 seconds

## Relational analysis of NS_B2_B1_A2_B1_A2_A2

### Relational analysis result of NS_B2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157954, upper bound: 86.5170950
time: 0.70 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -40.3924408, 89.3177872, -34.8615990, 77.1232071, -117.0526581, 123.8642120
1: -82.2222748, 133.1635284, -70.8923035, 114.7634583, -196.9857330, 204.0558319
2: -63.4041748, 128.4975586, -54.7062073, 110.7735291, -174.1777039, 183.2037659
3: -96.3773422, 153.8543091, -83.3374863, 132.4437408, -228.8210754, 237.1917877
4: -88.9540024, 147.1836395, -76.9158554, 126.6824036, -214.8525696, 223.7021332

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B2_B1_A2_B2_A1_A1

### Relational analysis result of NS_B2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5142981, upper bound: 86.5154596
time: 0.75 seconds

## Relational analysis of NS_B2_B1_A2_B2_A1_A2

### Relational analysis result of NS_B2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5135943, upper bound: 86.5144833
time: 0.69 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -42.3254776, 94.0175095, -34.7850990, 76.9285736, -118.7986526, 128.4636688
1: -86.3761139, 140.3014832, -70.6909714, 114.4633026, -200.8394165, 210.9924316
2: -66.5182800, 135.3114166, -54.5734444, 110.4774246, -176.9956970, 189.8848572
3: -101.1229095, 162.1529694, -83.1311493, 132.0962219, -233.2191315, 245.2841187
4: -93.2326126, 155.1610565, -76.7484436, 126.3468552, -218.8229523, 231.4573975

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B2_B1_A2_B2_A2_A1

### Relational analysis result of NS_B2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5144270, upper bound: 86.5154851
time: 0.80 seconds

## Relational analysis of NS_B2_B1_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_B2_A2_B1

### Relational analysis result of NS_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5167104, upper bound: 86.5178207
time: 0.67 seconds

## Relational analysis of NS_B2_B1_A2_B2_A2_B2

### Relational analysis result of NS_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5167104, upper bound: 86.5178760
time: 0.91 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -28.7743149, 63.4770966, -42.3586311, 93.4686356, -122.2429504, 105.8268967
1: -59.4856415, 94.8242722, -86.3193207, 139.4367523, -198.9223785, 181.1435852
2: -45.5844383, 92.2265701, -66.5352783, 134.6734619, -180.2578888, 158.7618256
3: -69.2623062, 110.0045013, -101.1380081, 161.1397095, -230.4019928, 211.1425171
4: -63.6406403, 105.0588760, -93.2940903, 154.1737518, -217.8143921, 198.1992340

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_B2_A1_A1_B1_B1

### Relational analysis result of NS_B2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5153446, upper bound: 86.5132257
time: 0.76 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_B2

### Relational analysis result of NS_B2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5158524, upper bound: 86.5140201
time: 0.74 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -28.7743149, 63.4770966, -45.0059853, 99.5970840, -128.3713989, 108.4830780
1: -59.4856415, 94.8242722, -91.9448090, 148.9105225, -208.3961639, 186.7690735
2: -45.5844383, 92.2265701, -70.7545853, 143.8456879, -189.4300995, 162.9811554
3: -69.2623062, 110.0045013, -107.5894241, 172.0835571, -241.3458405, 217.5939331
4: -63.6406403, 105.0588760, -99.1489639, 164.6640472, -228.3046875, 204.1235809

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B2_B2_A1_A1_B2_A1

### Relational analysis result of NS_B2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151703, upper bound: 86.5139589
time: 1.47 seconds

## Relational analysis of NS_B2_B2_A1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A1_A1_B2_A1

### Relational analysis result of NS_B2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5147692, upper bound: 86.5130935
time: 0.73 seconds

## Relational analysis of NS_B2_B2_A1_A1_B2_A2

### Relational analysis result of NS_B2_B2_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5141038, upper bound: 86.5127868
time: 0.77 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -43.1081581, 95.7331009, -43.0973015, 95.7074280, -138.2428436, 138.2577667
1: -87.9880371, 143.1307678, -87.9650879, 143.0918579, -231.0377960, 231.0539398
2: -67.7785492, 137.9443359, -67.7609253, 137.9069672, -205.6855164, 205.7052460
3: -103.0230560, 165.2603912, -102.9964371, 165.2152100, -268.0781250, 268.0975037
4: -94.9564819, 158.0632629, -94.9323807, 158.0202179, -251.9065704, 251.9260406

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A2_A1_B1_B1

### Relational analysis result of NS_B2_B2_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5095688, upper bound: 86.5105335
time: 0.70 seconds

## Relational analysis of NS_B2_B2_A2_A1_B1_B2

### Relational analysis result of NS_B2_B2_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5072322, upper bound: 86.5072323
time: 0.85 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -43.1081581, 95.7331009, -42.5727501, 94.0666504, -136.6433411, 137.6984100
1: -87.9880371, 143.1307678, -86.7530441, 140.3873291, -228.3753510, 229.8153534
2: -67.7785492, 137.9443359, -66.8667679, 135.5145721, -203.2931213, 204.7969666
3: -103.0230560, 165.2603912, -101.6655807, 162.2071991, -265.2131042, 266.7034912
4: -94.9564819, 158.0632629, -93.7577744, 155.1746063, -249.1543427, 250.6643524

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B2_A2_A1_B2_B1

### Relational analysis result of NS_B2_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5133210, upper bound: 86.5115262
time: 0.70 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A2_A1_B2_B1

### Relational analysis result of NS_B2_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5095688, upper bound: 86.5110981
time: 0.67 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2_B2

### Relational analysis result of NS_B2_B2_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5072322, upper bound: 86.5102461
time: 0.72 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -42.5835991, 94.0922241, -43.0973015, 95.7074280, -137.6834412, 136.6581726
1: -86.7759247, 140.4262085, -87.9650879, 143.0918579, -229.7991791, 228.3912964
2: -66.8843231, 135.5517731, -67.7609253, 137.9069672, -204.7766418, 203.3126984
3: -101.6921005, 162.2522888, -102.9964371, 165.2152100, -266.6840210, 265.2322998
4: -93.7818069, 155.2175446, -94.9323807, 158.0202179, -250.6448517, 249.1736450

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_A2_B1_A1

### Relational analysis result of NS_B2_B2_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5134830, upper bound: 86.5137310
time: 0.71 seconds

## Relational analysis of NS_B2_B2_A2_A2_B1_A2

### Relational analysis result of NS_B2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145660, upper bound: 86.5145660
time: 1.01 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -42.5835991, 94.0922241, -42.5727501, 94.0666504, -136.1450653, 136.1599121
1: -86.7759247, 140.4262085, -86.7530441, 140.3873291, -227.1632538, 227.1792450
2: -66.8843231, 135.5517731, -66.8667679, 135.5145721, -202.3988953, 202.4185486
3: -101.6921005, 162.2522888, -101.6655807, 162.2071991, -263.8992920, 263.9178467
4: -93.7818069, 155.2175446, -93.7577744, 155.1746063, -248.0549622, 248.0742645

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_A2_B2_A1

### Relational analysis result of NS_B2_B2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5134830, upper bound: 86.5137310
time: 1.03 seconds

## Relational analysis of NS_B2_B2_A2_A2_B2_A2

### Relational analysis result of NS_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5145660, upper bound: 86.5145660
time: 0.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.90 seconds
NS_B1_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5149016, upper bound: 86.5149016
NS_B1_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5149016, upper bound: 86.5154611
NS_B1_B2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5148390, upper bound: 86.5125485
NS_B1_B2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5149776, upper bound: 86.5118547
NS_B1_B2_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5178436, upper bound: 86.5156933
NS_B1_B2_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5178201, upper bound: 86.5155240
NS_B1_B2_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5148109, upper bound: 86.5147187
NS_B1_B2_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5148109, upper bound: 86.5152786
NS_B1_B2_A2_A1_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5130848, upper bound: 86.5116717
NS_B1_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5162302, upper bound: 86.5151061
NS_B1_B2_A2_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5094315, upper bound: 86.5098741
NS_B1_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5157376, upper bound: 86.5150166
NS_B1_B2_A2_A2_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5130893, upper bound: 86.5136584
NS_B1_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5160864, upper bound: 86.5151061
NS_B2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5184728, upper bound: 86.5191232
NS_B2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5184603, upper bound: 86.5192452
NS_B2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5184728, upper bound: 86.5191232
NS_B2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5184603, upper bound: 86.5192452
NS_B2_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5182644, upper bound: 86.5183953
NS_B2_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5183700, upper bound: 86.5183700
NS_B2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5183953, upper bound: 86.5182644
NS_B2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5183700, upper bound: 86.5183700
NS_B2_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5156877, upper bound: 86.5175476
NS_B2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5157195, upper bound: 86.5164306
NS_B2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5157088, upper bound: 86.5175474
NS_B2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5157954, upper bound: 86.5170950
NS_B2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5142981, upper bound: 86.5154596
NS_B2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5135943, upper bound: 86.5144833
NS_B2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5167104, upper bound: 86.5178207
NS_B2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5167104, upper bound: 86.5178760
NS_B2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5153446, upper bound: 86.5132257
NS_B2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5158524, upper bound: 86.5140201
NS_B2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5147692, upper bound: 86.5130935
NS_B2_B2_A1_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5141038, upper bound: 86.5127868
NS_B2_B2_A2_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5095688, upper bound: 86.5105335
NS_B2_B2_A2_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5072322, upper bound: 86.5072323
NS_B2_B2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5095688, upper bound: 86.5110981
NS_B2_B2_A2_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5072322, upper bound: 86.5102461
NS_B2_B2_A2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5134830, upper bound: 86.5137310
NS_B2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5145660, upper bound: 86.5145660
NS_B2_B2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5134830, upper bound: 86.5137310
NS_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 0, lower bound: -86.5145660, upper bound: 86.5145660

## BFS NS instance: NS_B1_B1_A1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -31.0056629, 67.5940628, -27.6562996, 60.4842339, -91.4898911, 95.2503662
1: -63.9520798, 100.9623337, -57.0627022, 90.2550125, -154.2070923, 158.0250092
2: -49.0105057, 98.4064178, -43.7578621, 87.9879913, -136.9985046, 142.1642761
3: -74.6618500, 117.1084137, -66.5309448, 104.7701645, -179.4320068, 183.6393433
4: -68.5695877, 111.9353485, -61.1693115, 100.0778275, -168.6473999, 173.1046600

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_B1_B1_A1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -31.4691200, 69.0881424, -27.6562996, 60.4842339, -91.9533539, 96.7444458
1: -65.0073929, 103.2511368, -57.0627022, 90.2550125, -155.2624054, 160.3138123
2: -49.7990417, 100.4536362, -43.7578621, 87.9879913, -137.7870331, 144.2114716
3: -75.8150177, 119.7603607, -66.5309448, 104.7701645, -180.5851746, 186.2913055
4: -69.5918655, 114.3924789, -61.1693115, 100.0778275, -169.6696777, 175.5617828

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

## BFS NS instance: NS_B1_B2_A1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -25.2481079, 55.3948822, -37.4428406, 81.1585693, -106.4066772, 92.8377151
1: -51.7680435, 82.6171417, -76.1406479, 120.8200989, -172.5881348, 158.7577820
2: -39.8541298, 80.3703690, -58.6763000, 117.8571854, -157.7112885, 139.0466614
3: -60.5615883, 95.6949997, -89.2204361, 140.4093018, -200.9708862, 184.9154358
4: -55.8339844, 91.5198669, -82.5170822, 133.8450012, -189.6789856, 174.0369263

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B2_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B2_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B2_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B2_A1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_B2_A1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -25.5041485, 56.0015030, -37.1618843, 80.5563965, -106.0605392, 93.1633835
1: -52.3322334, 83.5410156, -75.5508041, 119.9143219, -172.2465515, 159.0917816
2: -40.2744904, 81.2849808, -58.2294464, 116.9652634, -157.2397461, 139.5144348
3: -61.2096443, 96.7680893, -88.5397949, 139.3504791, -200.5601044, 185.3078308
4: -56.4041252, 92.5663605, -81.8976822, 132.8412323, -189.2453613, 174.4640198

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B2_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B2_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_B2_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B2_A1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_B2_A1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -28.1213169, 61.9262276, -38.1475983, 82.8817978, -111.0031128, 100.0738220
1: -58.0048943, 92.4405899, -77.7305908, 123.4044495, -181.4093018, 170.1711273
2: -44.5087090, 89.8918304, -59.8360825, 120.3889465, -164.8976593, 149.7279053
3: -67.6335144, 107.1468048, -90.9873352, 143.4753418, -211.1088562, 198.1341400
4: -62.1994514, 102.3908691, -84.0758667, 136.7279510, -198.9273834, 186.4667358

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B2_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B2_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B2_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B2_A1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_B2_A1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -28.4032402, 62.5878601, -37.8593178, 82.2646255, -110.6678543, 100.4471741
1: -58.6074982, 93.4663925, -77.1258316, 122.4769821, -181.0844727, 170.5922241
2: -44.9677010, 90.9019623, -59.3778076, 119.4757767, -164.4434357, 150.2797699
3: -68.3365555, 108.3543167, -90.2892075, 142.3920288, -210.7285767, 198.6435242
4: -62.8274155, 103.5420685, -83.4403839, 135.6997833, -198.5271912, 186.9824371

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B2_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B2_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_B2_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B2_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_B2_A1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -31.0056629, 67.5940628, -37.5801201, 81.5999985, -112.6056595, 105.1741791
1: -63.9520798, 100.9623337, -76.5142136, 121.4675140, -185.4195862, 177.4765320
2: -49.0105057, 98.4064178, -58.9239960, 118.5308914, -167.5413971, 157.3304138
3: -74.6618500, 117.1084137, -89.5940399, 141.2862091, -215.9480591, 206.7024078
4: -68.5695877, 111.9353485, -82.8222351, 134.5755768, -203.1451721, 194.7575836

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_B1_B2_A1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -31.4691200, 69.0881424, -37.5845604, 81.6109161, -113.0800323, 106.6726990
1: -65.0073929, 103.2511368, -76.5249786, 121.4846649, -186.4920349, 179.7760773
2: -49.7990417, 100.4536362, -58.9313622, 118.5472946, -168.3463287, 159.3849945
3: -75.8150177, 119.7603607, -89.6054001, 141.3057098, -217.1207275, 209.3657379
4: -69.5918655, 114.3924789, -82.8319550, 134.5943146, -204.1861725, 197.2244110

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

## BFS NS instance: NS_B1_B2_A2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -42.0200920, 92.7918472, -37.9622498, 82.5064011, -124.2662354, 130.3387909
1: -85.5151672, 138.5058594, -77.3621826, 122.8459702, -208.3611450, 215.8680420
2: -65.9670715, 133.6212311, -59.5481071, 119.8418503, -185.8089294, 193.1693268
3: -100.3071136, 159.9564972, -90.5491867, 142.8300476, -243.1371613, 250.5056610
4: -92.5397949, 153.0373077, -83.6675339, 136.1157990, -228.2357941, 236.0367279

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_B2_A2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B2_A2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_B2_A2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B2_A2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_B2_A2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -45.2729874, 99.3887253, -37.8761902, 82.2366333, -127.1843872, 136.8542938
1: -92.0707245, 148.3374023, -77.1387405, 122.4256134, -214.4963379, 225.4761047
2: -71.0125122, 143.2384491, -59.3954010, 119.4734497, -190.4859619, 202.6338501
3: -108.0956116, 171.3131409, -90.3126297, 142.4047241, -250.4636841, 261.6257629
4: -99.7398682, 163.9046631, -83.4748611, 135.6372833, -234.7479095, 246.7400360

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B2_A2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_B2_A2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_B2_A2_A2_A1_A2_B1

### Relational analysis result of NS_B1_B2_A2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5099225, upper bound: 86.5105300
time: 0.93 seconds

## Relational analysis of NS_B1_B2_A2_A2_A1_A2_B2

### Relational analysis result of NS_B1_B2_A2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5099225, upper bound: 86.5150166
time: 0.71 seconds

## BFS NS instance: NS_B1_B2_A2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -45.3652916, 99.6985397, -37.5845604, 81.6109161, -126.6779861, 136.8905792
1: -92.3600693, 148.8137665, -76.5249786, 121.4846649, -213.8447113, 225.3387299
2: -71.1947632, 143.7209473, -58.9313622, 118.5472946, -189.7420654, 202.6522980
3: -108.3835831, 171.8863525, -89.6054001, 141.3057098, -249.6892700, 261.4917297
4: -99.9376831, 164.4867554, -82.8319550, 134.5943146, -233.9727478, 246.7162476

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_B2_A2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B2_A2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_B2_A2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_B2_A2_A2_A2_A2_B1

### Relational analysis result of NS_B1_B2_A2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5117663, upper bound: 86.5110990
time: 0.68 seconds

## Relational analysis of NS_B1_B2_A2_A2_A2_A2_B2

### Relational analysis result of NS_B1_B2_A2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5117663, upper bound: 86.5151061
time: 0.69 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -25.8996620, 57.2254372, -24.8325138, 54.9871826, -80.8868408, 82.0579529
1: -53.0910454, 85.4786758, -50.9042320, 82.1095734, -135.2006073, 136.3828888
2: -40.8890762, 82.5770340, -39.2062607, 79.2653961, -120.1544724, 121.7832947
3: -62.1785469, 98.7102737, -59.6130447, 94.7958984, -156.9744110, 158.3233185
4: -57.3680992, 94.3456421, -55.0092812, 90.6218491, -147.9899445, 149.3549194

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_B2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5181982, upper bound: 86.5189091
time: 0.63 seconds

## Relational analysis of NS_B2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_B2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5188585, upper bound: 86.5188505
time: 0.86 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -25.8599644, 57.1162605, -26.6515274, 59.4921684, -85.3521347, 83.7677917
1: -52.9793701, 85.3091202, -54.8183479, 89.0429993, -142.0223541, 140.1274414
2: -40.8167839, 82.4143524, -42.1476860, 85.8687134, -126.6855011, 124.5620422
3: -62.0709038, 98.5163574, -64.1015701, 102.9113312, -164.9822235, 162.6179199
4: -57.2830315, 94.1601639, -59.0593567, 98.2843475, -155.5673828, 153.2195129

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_B2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5200658, upper bound: 86.5202793
time: 0.71 seconds

## Relational analysis of NS_B2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_B2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5200658, upper bound: 86.5202793
time: 0.65 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -34.8718910, 77.1468735, -24.8325138, 54.9871826, -89.8590698, 101.9793854
1: -70.9135361, 114.7993088, -50.9042320, 82.1095734, -153.0230865, 165.7035370
2: -54.7225113, 110.8079071, -39.2062607, 79.2653961, -133.9878998, 150.0141449
3: -83.3621368, 132.4853821, -59.6130447, 94.7958984, -178.1580353, 192.0984192
4: -76.9383392, 126.7219772, -55.0092812, 90.6218491, -167.5601807, 181.7312164

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A1_B1_A2_B1_B1

### Relational analysis result of NS_B2_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5184509, upper bound: 86.5189080
time: 0.70 seconds

## Relational analysis of NS_B2_B1_A1_B1_A2_B1_B2

### Relational analysis result of NS_B2_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5183525, upper bound: 86.5186661
time: 1.30 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -34.7952499, 76.9519730, -26.6515274, 59.4921684, -94.2874146, 103.6034927
1: -70.7118530, 114.4987259, -54.8183479, 89.0429993, -159.7548370, 169.3170471
2: -54.5895424, 110.5113754, -42.1476860, 85.8687134, -140.4582367, 152.6590576
3: -83.1554565, 132.1373596, -64.1015701, 102.9113312, -186.0667572, 196.2389221
4: -76.7706375, 126.3859482, -59.0593567, 98.2843475, -175.0549927, 185.4453125

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_B2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5183842, upper bound: 86.5192440
time: 0.96 seconds

## Relational analysis of NS_B2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_B2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5183842, upper bound: 86.5192452
time: 1.39 seconds

## BFS NS instance: NS_B2_B1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -24.8435326, 55.0127487, -34.8615990, 77.1232071, -101.9667358, 89.8743439
1: -50.9271774, 82.1485672, -70.8923035, 114.7634583, -165.6906433, 153.0408630
2: -39.2237816, 79.3026886, -54.7062073, 110.7735291, -149.9973145, 134.0088959
3: -59.6396294, 94.8412476, -83.3374863, 132.4437408, -192.0833740, 178.1787415
4: -55.0333939, 90.6646805, -76.9158554, 126.6824036, -181.7157898, 167.5805359

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A1

### Relational analysis result of NS_B2_B1_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5178322, upper bound: 86.5176973
time: 0.68 seconds

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A1

### Relational analysis result of NS_B2_B1_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5189080, upper bound: 86.5184509
time: 0.67 seconds

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A2

### Relational analysis result of NS_B2_B1_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5186661, upper bound: 86.5183525
time: 0.74 seconds

## BFS NS instance: NS_B2_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -26.6619759, 59.5165520, -34.7850990, 76.9285736, -103.5905457, 94.3016510
1: -54.8399620, 89.0801697, -70.6909714, 114.4633026, -169.3032532, 159.7711029
2: -42.1642952, 85.9041443, -54.5734444, 110.4774246, -152.6417236, 140.4775848
3: -64.1267319, 102.9544754, -83.1311493, 132.0962219, -196.2229462, 186.0856323
4: -59.0822067, 98.3251266, -76.7484436, 126.3468552, -185.4290619, 175.0735626

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A1_B2_A1_A2_A1

### Relational analysis result of NS_B2_B1_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5191001, upper bound: 86.5184564
time: 0.76 seconds

## Relational analysis of NS_B2_B1_A1_B2_A1_A2_A2

### Relational analysis result of NS_B2_B1_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5191001, upper bound: 86.5183757
time: 0.65 seconds

## BFS NS instance: NS_B2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -34.8718910, 77.1468735, -33.6381950, 74.5331650, -109.1147842, 110.4961853
1: -70.9135361, 114.7993088, -68.3656464, 110.8520889, -181.7656250, 183.1649475
2: -54.7225113, 110.8079071, -52.7665825, 106.9071579, -161.6296692, 163.5744934
3: -83.3621368, 132.4853821, -80.3928909, 127.8606110, -211.2227478, 212.8782654
4: -76.9383392, 126.7219772, -74.2124176, 122.3466263, -198.9323425, 200.5885315

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A1_B2_A2_B1_B1

### Relational analysis result of NS_B2_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5183574, upper bound: 86.5180280
time: 0.77 seconds

## Relational analysis of NS_B2_B1_A1_B2_A2_B1_B2

### Relational analysis result of NS_B2_B1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5182092, upper bound: 86.5176963
time: 1.06 seconds

## BFS NS instance: NS_B2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -34.7952499, 76.9519730, -35.5835419, 79.2975388, -113.7741776, 112.2611847
1: -70.7118530, 114.4987259, -72.5290070, 118.1376572, -188.8494720, 187.0277405
2: -54.5895424, 110.5113754, -55.9086151, 113.8540878, -168.4436188, 166.4199829
3: -83.1554565, 132.1373596, -85.1687546, 136.3669434, -219.5223999, 217.3060913
4: -76.7706375, 126.3859482, -78.5361633, 130.4347076, -206.7838287, 204.6119690

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_B1_A1_B2_A2_B2_B1

### Relational analysis result of NS_B2_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5160075, upper bound: 86.5161019
time: 1.25 seconds

## Relational analysis of NS_B2_B1_A1_B2_A2_B2_B2

### Relational analysis result of NS_B2_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151787, upper bound: 86.5151787
time: 0.63 seconds

## BFS NS instance: NS_B2_B1_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -38.8508072, 85.7485352, -25.8438511, 57.0997963, -95.9327469, 111.5923615
1: -78.6569061, 127.8892288, -52.9698792, 85.2893066, -163.9462128, 180.8590851
2: -60.8401604, 123.0707474, -40.7991066, 82.3903580, -143.2305145, 163.8698578
3: -92.5203629, 147.4676819, -62.0411263, 98.4881668, -191.0085297, 209.5088043
4: -85.5891037, 141.0927582, -57.2456818, 94.1336670, -179.5175629, 198.3384399

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_B1_A1_A1_B1

### Relational analysis result of NS_B2_B1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5152776, upper bound: 86.5174738
time: 0.61 seconds

## Relational analysis of NS_B2_B1_A2_B1_A1_A1_B2

### Relational analysis result of NS_B2_B1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5152776, upper bound: 86.5175476
time: 0.65 seconds

## BFS NS instance: NS_B2_B1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -39.0329247, 86.1450424, -25.6521988, 56.6580925, -95.6744690, 111.7972183
1: -79.0181503, 128.5476227, -52.5393867, 84.6283569, -163.6464996, 181.0869751
2: -61.1153297, 123.7182770, -40.4851570, 81.7367096, -142.8520203, 164.2033997
3: -92.9472122, 148.2513428, -61.5620384, 97.7107773, -190.6579895, 209.8133850
4: -85.9852219, 141.8006744, -56.8230743, 93.3882217, -179.1702728, 198.6237488

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_B1_A1_A2_B1

### Relational analysis result of NS_B2_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5152297, upper bound: 86.5162002
time: 0.75 seconds

## Relational analysis of NS_B2_B1_A2_B1_A1_A2_B2

### Relational analysis result of NS_B2_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5152297, upper bound: 86.5164306
time: 1.02 seconds

## BFS NS instance: NS_B2_B1_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -40.6585541, 90.1947937, -25.8015461, 56.9849434, -97.6310501, 115.9963379
1: -82.5580673, 134.6802368, -52.8527565, 85.1111984, -167.6692657, 187.5329742
2: -63.7647705, 129.5316620, -40.7226906, 82.2193756, -145.9841156, 170.2543488
3: -96.9672089, 155.3753967, -61.9271088, 98.2834930, -195.2506409, 217.3025055
4: -89.5934906, 148.6343994, -57.1549416, 93.9387970, -183.3488922, 205.7893372

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A2_B1_A2_A1_A1

### Relational analysis result of NS_B2_B1_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157088, upper bound: 86.5175238
time: 0.62 seconds

## Relational analysis of NS_B2_B1_A2_B1_A2_A1_A2

### Relational analysis result of NS_B2_B1_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5154486, upper bound: 86.5172735
time: 0.73 seconds

## BFS NS instance: NS_B2_B1_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -41.0314255, 90.9678497, -25.6301651, 56.5865250, -97.6061935, 116.5980148
1: -83.2942047, 135.8736725, -52.4641533, 84.5152206, -167.8094177, 188.3377991
2: -64.3385544, 130.6892853, -40.4406090, 81.6286545, -145.9671783, 171.1298676
3: -97.8414993, 156.7398987, -61.4969673, 97.5834198, -195.4248657, 218.2368622
4: -90.3988953, 149.9468689, -56.7765808, 93.2659836, -183.4844208, 206.7234497

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A2_B1_A2_A2_A1

### Relational analysis result of NS_B2_B1_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157954, upper bound: 86.5170950
time: 0.75 seconds

## Relational analysis of NS_B2_B1_A2_B1_A2_A2_A2

### Relational analysis result of NS_B2_B1_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157879, upper bound: 86.5170950
time: 0.64 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -40.2638702, 89.5542450, -34.2125740, 75.7364044, -115.5172882, 123.3670731
1: -82.0142288, 133.7790375, -69.5399017, 112.6793976, -194.6936340, 203.2417145
2: -63.2365036, 128.8125458, -53.6788559, 108.7238770, -171.9603882, 182.4913940
3: -96.1096039, 154.4377441, -81.7689972, 130.0149841, -226.1200104, 236.2067413
4: -88.6934814, 147.6777649, -75.4872665, 124.3710480, -212.2273407, 222.5567627

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_B1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A2_B2_A1_A1_A1

### Relational analysis result of NS_B2_B1_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5123203, upper bound: 86.5147120
time: 0.69 seconds

## Relational analysis of NS_B2_B1_A2_B2_A1_A1_A2

### Relational analysis result of NS_B2_B1_A2_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5123591, upper bound: 86.5131220
time: 1.12 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -39.8079796, 88.0827713, -34.6721344, 76.7309341, -116.0749130, 122.4260635
1: -80.9704590, 131.3232269, -70.4917755, 114.1785431, -195.1489563, 201.8149719
2: -62.4623489, 126.6869278, -54.4016380, 110.1946869, -172.6570282, 181.0885620
3: -94.9583282, 151.7104492, -82.8798294, 131.7595062, -226.7178345, 234.5902710
4: -87.6681519, 145.1218109, -76.4972382, 126.0253983, -212.9103241, 221.1929932

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_B1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A2_B2_A1_A2_A1

### Relational analysis result of NS_B2_B1_A2_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5115262, upper bound: 86.5135824
time: 0.75 seconds

## Relational analysis of NS_B2_B1_A2_B2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_B2_A1_A2_B1

### Relational analysis result of NS_B2_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5135943, upper bound: 86.5144833
time: 0.99 seconds

## Relational analysis of NS_B2_B1_A2_B2_A1_A2_B2

### Relational analysis result of NS_B2_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5135943, upper bound: 86.5144833
time: 0.70 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -42.3151283, 93.9925003, -33.6381950, 74.5331650, -116.3823853, 127.2806473
1: -86.3554688, 140.2626648, -68.3656464, 110.8520889, -197.2075500, 208.6283112
2: -66.5021362, 135.2767181, -52.7665825, 106.9071579, -173.4093018, 188.0433044
3: -101.0980988, 162.1093903, -80.3928909, 127.8606110, -228.9587097, 242.5022583
4: -93.2101440, 155.1199951, -74.2124176, 122.3466263, -214.7672577, 228.8526001

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_B2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5150897, upper bound: 86.5171544
time: 0.63 seconds

## Relational analysis of NS_B2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_B2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151879, upper bound: 86.5166031
time: 0.90 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -42.3235474, 94.0128632, -35.5835419, 79.2975388, -121.1254654, 129.2614288
1: -86.3722687, 140.2942352, -72.5290070, 118.1376572, -204.5098724, 212.8232422
2: -66.5152817, 135.3049774, -55.9086151, 113.8540878, -180.3693695, 191.2135620
3: -101.1183014, 162.1448669, -85.1687546, 136.3669434, -237.4852448, 247.3136292
4: -93.2284241, 155.1533966, -78.5361633, 130.4347076, -222.8028412, 233.2449646

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_B2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5150897, upper bound: 86.5171719
time: 0.63 seconds

## Relational analysis of NS_B2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_B2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151879, upper bound: 86.5168705
time: 0.75 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -28.1737823, 62.2241096, -42.3204308, 93.8916550, -122.0654373, 104.5189514
1: -58.2508698, 92.9594727, -86.3132935, 140.3218842, -198.5727386, 179.2727661
2: -44.6403580, 90.3866043, -66.5180511, 135.2680511, -179.9084167, 156.9046478
3: -67.8143921, 107.8376846, -101.0954132, 162.0272369, -229.8416290, 208.9331055
4: -62.3207855, 102.9905014, -93.2276764, 154.9666443, -217.2874146, 196.0115967

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_A1_B1_B1_A1

### Relational analysis result of NS_B2_B2_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5170052, upper bound: 86.5152818
time: 0.94 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_B1_A2

### Relational analysis result of NS_B2_B2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5164346, upper bound: 86.5152710
time: 0.69 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -28.5803242, 63.0612679, -41.8480034, 92.3833160, -120.9636383, 104.9047318
1: -59.0650597, 94.1975250, -85.2240372, 137.8187561, -196.8838043, 179.4215698
2: -45.2713509, 91.6095352, -65.7100830, 133.0742645, -178.3455963, 157.3195953
3: -68.7877579, 109.2669678, -99.8978195, 159.2503357, -228.0380859, 209.1647491
4: -63.2127380, 104.3585281, -92.1685104, 152.3555145, -215.5682526, 196.3830414

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B2_B2_A1_A1_B1_B2_A1

### Relational analysis result of NS_B2_B2_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5163228, upper bound: 86.5153009
time: 0.76 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_B2_A2

### Relational analysis result of NS_B2_B2_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5163228, upper bound: 86.5153260
time: 0.71 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -28.3800182, 62.6051598, -44.9489403, 99.4563141, -127.8363342, 107.5540924
1: -58.6082268, 93.5178146, -91.8285217, 148.6927795, -207.3009644, 185.3463287
2: -44.9437447, 90.9242020, -70.6646271, 143.6456451, -188.5893860, 161.5888214
3: -68.2776871, 108.4670181, -107.4512253, 171.8373260, -240.1150208, 215.9182281
4: -62.7703247, 103.5859680, -99.0236969, 164.4291840, -227.1994934, 202.5278168

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B2_B2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B2_B2_A1_A1_B2_A1_A1

### Relational analysis result of NS_B2_B2_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143369, upper bound: 86.5128703
time: 0.68 seconds

## Relational analysis of NS_B2_B2_A1_A1_B2_A1_A2

### Relational analysis result of NS_B2_B2_A1_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5141217, upper bound: 86.5128987
time: 1.27 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -42.3381729, 93.5244446, -43.0142670, 95.5155182, -137.2310791, 135.9968109
1: -86.2220154, 139.5653534, -87.7761841, 142.8025513, -228.9160309, 227.3415070
2: -66.4798203, 134.7054291, -67.6232224, 137.6199951, -204.0327911, 202.3286438
3: -101.0780792, 161.2415619, -102.7873230, 164.8749542, -265.6794739, 263.9861450
4: -93.2408371, 154.2395782, -94.7493973, 157.6914673, -249.7058258, 247.9806366

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_A2_B1_A2_B1

### Relational analysis result of NS_B2_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5147881, upper bound: 86.5145565
time: 0.74 seconds

## Relational analysis of NS_B2_B2_A2_A2_B1_A2_B2

### Relational analysis result of NS_B2_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5147881, upper bound: 86.5149630
time: 0.76 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -42.3381729, 93.5244446, -42.4796410, 93.8518372, -135.6696625, 135.4881287
1: -86.2220154, 139.5653534, -86.5430984, 140.0611267, -226.2831421, 226.1084290
2: -66.4798203, 134.7054291, -66.7133026, 135.1940918, -201.6739197, 201.4187317
3: -101.0780792, 161.2415619, -101.4329605, 161.8238678, -262.8994141, 262.6744080
4: -93.2408371, 154.2395782, -93.5526123, 154.8047180, -247.0767212, 246.8581238

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_B1

### Relational analysis result of NS_B2_B2_A2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5137310, upper bound: 86.5134830
time: 0.80 seconds

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_B2

### Relational analysis result of NS_B2_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5137310, upper bound: 86.5145660
time: 0.67 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.35 seconds
NS_B1_B2_A2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5099225, upper bound: 86.5105300
NS_B1_B2_A2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5099225, upper bound: 86.5150166
NS_B1_B2_A2_A2_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5117663, upper bound: 86.5110990
NS_B1_B2_A2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5117663, upper bound: 86.5151061
NS_B2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5181982, upper bound: 86.5189091
NS_B2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5188585, upper bound: 86.5188505
NS_B2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5200658, upper bound: 86.5202793
NS_B2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5200658, upper bound: 86.5202793
NS_B2_B1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5184509, upper bound: 86.5189080
NS_B2_B1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5183525, upper bound: 86.5186661
NS_B2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5183842, upper bound: 86.5192440
NS_B2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5183842, upper bound: 86.5192452
NS_B2_B1_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5189080, upper bound: 86.5184509
NS_B2_B1_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5186661, upper bound: 86.5183525
NS_B2_B1_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5191001, upper bound: 86.5184564
NS_B2_B1_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5191001, upper bound: 86.5183757
NS_B2_B1_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5183574, upper bound: 86.5180280
NS_B2_B1_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5182092, upper bound: 86.5176963
NS_B2_B1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5160075, upper bound: 86.5161019
NS_B2_B1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5151787, upper bound: 86.5151787
NS_B2_B1_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5152776, upper bound: 86.5174738
NS_B2_B1_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5152776, upper bound: 86.5175476
NS_B2_B1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5152297, upper bound: 86.5162002
NS_B2_B1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5152297, upper bound: 86.5164306
NS_B2_B1_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5157088, upper bound: 86.5175238
NS_B2_B1_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5154486, upper bound: 86.5172735
NS_B2_B1_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5157954, upper bound: 86.5170950
NS_B2_B1_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5157879, upper bound: 86.5170950
NS_B2_B1_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5123203, upper bound: 86.5147120
NS_B2_B1_A2_B2_A1_A1_A2, status: Status.VERIFIED, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5123591, upper bound: 86.5131220
NS_B2_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5135943, upper bound: 86.5144833
NS_B2_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5135943, upper bound: 86.5144833
NS_B2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5150897, upper bound: 86.5171544
NS_B2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5151879, upper bound: 86.5166031
NS_B2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5150897, upper bound: 86.5171719
NS_B2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5151879, upper bound: 86.5168705
NS_B2_B2_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5170052, upper bound: 86.5152818
NS_B2_B2_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5164346, upper bound: 86.5152710
NS_B2_B2_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5163228, upper bound: 86.5153009
NS_B2_B2_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5163228, upper bound: 86.5153260
NS_B2_B2_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5143369, upper bound: 86.5128703
NS_B2_B2_A1_A1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5141217, upper bound: 86.5128987
NS_B2_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5147881, upper bound: 86.5145565
NS_B2_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5147881, upper bound: 86.5149630
NS_B2_B2_A2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5137310, upper bound: 86.5134830
NS_B2_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.35
Output dim: 0, lower bound: -86.5137310, upper bound: 86.5145660

## BFS NS instance: NS_B1_B2_A2_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -45.2729874, 99.3887253, -37.8143005, 82.0973358, -127.0441742, 136.7923737
1: -92.0707245, 148.3374023, -76.9957809, 122.2161026, -214.2868195, 225.3331757
2: -71.0125122, 143.2384491, -59.2935371, 119.2544098, -190.2669220, 202.5319824
3: -108.0956116, 171.3131409, -90.1567764, 142.1452179, -250.2056122, 261.4699097
4: -99.7398682, 163.9046631, -83.3397827, 135.3969421, -234.5048828, 246.6042023

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B2_A2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_B2_A2_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -45.3652916, 99.6985397, -37.5218811, 81.4701233, -126.5362930, 136.8278351
1: -92.3600693, 148.8137665, -76.3807297, 121.2729340, -213.6329498, 225.1944885
2: -71.1947632, 143.7209473, -58.8283234, 118.3260880, -189.5208435, 202.5492706
3: -108.3835831, 171.8863525, -89.4478455, 141.0436249, -249.4271851, 261.3341980
4: -99.9376831, 164.4867554, -82.6951370, 134.3515472, -233.7272644, 246.5786591

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_B2_A2_A2_A2_A2_B2_A1

### Relational analysis result of NS_B1_B2_A2_A2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5103501, upper bound: 86.5145621
time: 1.34 seconds

## Relational analysis of NS_B1_B2_A2_A2_A2_A2_B2_A2

### Relational analysis result of NS_B1_B2_A2_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5103501, upper bound: 86.5150166
time: 1.24 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -24.9929028, 55.1936455, -24.7947807, 54.9026337, -79.8955383, 79.9884262
1: -51.1022110, 82.4255676, -50.8208466, 81.9825592, -133.0847778, 133.2464142
2: -39.4201202, 79.5396347, -39.1448898, 79.1387329, -118.5588531, 118.6845245
3: -59.9365768, 95.1197815, -59.5194817, 94.6461639, -154.5827179, 154.6392670
4: -55.3781166, 90.9076996, -54.9264069, 90.4786148, -145.8567047, 145.8341064

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_B2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5181982, upper bound: 86.5186575
time: 0.75 seconds

## Relational analysis of NS_B2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_B2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5181982, upper bound: 86.5188480
time: 0.76 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -25.1023197, 55.4789200, -24.6229420, 54.5028534, -79.6051636, 80.1018600
1: -51.3078156, 82.8974152, -50.4295845, 81.3858566, -132.6936646, 133.3269958
2: -39.5814896, 79.9709396, -38.8616714, 78.5468369, -118.1283264, 118.8326111
3: -60.1899796, 95.6519241, -59.0867767, 93.9430542, -154.1330109, 154.7387085
4: -55.6106911, 91.4047470, -54.5465927, 89.8019714, -145.4126434, 145.9513397

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_B2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5188585, upper bound: 86.5186602
time: 0.83 seconds

## Relational analysis of NS_B2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_B2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5188585, upper bound: 86.5188505
time: 0.79 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -24.8435326, 55.0127487, -26.6515274, 59.4921684, -84.3357010, 81.6642761
1: -50.9271774, 82.1485672, -54.8183479, 89.0429993, -139.9701843, 136.9668732
2: -39.2237816, 79.3026886, -42.1476860, 85.8687134, -125.0924911, 121.4503784
3: -59.6396294, 94.8412476, -64.1015701, 102.9113312, -162.5509644, 158.9428101
4: -55.0333939, 90.6646805, -59.0593567, 98.2843475, -153.3177490, 149.7240295

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A1_B1_A1_B2_A1_A1

### Relational analysis result of NS_B2_B1_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5181932, upper bound: 86.5192403
time: 0.72 seconds

## Relational analysis of NS_B2_B1_A1_B1_A1_B2_A1_A2

### Relational analysis result of NS_B2_B1_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5184326, upper bound: 86.5188585
time: 0.79 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -26.6619759, 59.5165520, -26.6515274, 59.4921684, -86.1541443, 86.1680756
1: -54.8399620, 89.0801697, -54.8183479, 89.0429993, -143.8829346, 143.8984680
2: -42.1642952, 85.9041443, -42.1476860, 85.8687134, -128.0330048, 128.0518341
3: -64.1267319, 102.9544754, -64.1015701, 102.9113312, -167.0380402, 167.0560455
4: -59.0822067, 98.3251266, -59.0593567, 98.2843475, -157.3665466, 157.3844757

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A1_B1_A1_B2_A2_A1

### Relational analysis result of NS_B2_B1_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5181932, upper bound: 86.5192925
time: 0.67 seconds

## Relational analysis of NS_B2_B1_A1_B1_A1_B2_A2_A2

### Relational analysis result of NS_B2_B1_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5184326, upper bound: 86.5184326
time: 1.11 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -34.6841049, 76.7415085, -24.4840832, 54.2308731, -88.9149704, 101.2255936
1: -70.5182114, 114.1923141, -50.1643486, 80.9797668, -151.4979858, 164.3566589
2: -54.4218407, 110.2171631, -38.6499252, 78.1625061, -132.5843048, 148.8670807
3: -82.9073639, 131.7827759, -58.7663002, 93.4936752, -176.4010315, 190.5490723
4: -76.5263824, 126.0537949, -54.2462311, 89.3761597, -165.9025116, 180.3000183

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_B1_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_B2_B1_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5179793, upper bound: 86.5189080
time: 0.69 seconds

## Relational analysis of NS_B2_B1_A1_B1_A2_B1_B1_A2

### Relational analysis result of NS_B2_B1_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5179793, upper bound: 86.5189080
time: 1.07 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -34.1875801, 75.5207901, -24.4690971, 54.0862122, -88.2737885, 99.9898834
1: -69.4397964, 112.3093414, -50.2423820, 80.5766983, -150.0164948, 162.5516968
2: -53.6178665, 108.4025574, -38.6295662, 77.8388214, -131.4566650, 147.0321198
3: -81.6805954, 129.5991516, -58.7643585, 92.9990005, -174.6795959, 188.3635101
4: -75.4319077, 123.9631348, -54.1233215, 89.0440369, -164.4759521, 178.0864563

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_B1_A1_B1_A2_B1_B2_B1

### Relational analysis result of NS_B2_B1_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5160084, upper bound: 86.5164144
time: 0.62 seconds

## Relational analysis of NS_B2_B1_A1_B1_A2_B1_B2_B2

### Relational analysis result of NS_B2_B1_A1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5160730, upper bound: 86.5169680
time: 0.67 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -33.6487007, 74.5571442, -26.6515274, 59.4921684, -93.1408691, 101.2086716
1: -68.3872757, 110.8884888, -54.8183479, 89.0429993, -157.4302368, 165.7067871
2: -52.7832260, 106.9420242, -42.1476860, 85.8687134, -138.6519470, 149.0897064
3: -80.4180450, 127.9029007, -64.1015701, 102.9113312, -183.3293610, 192.0044403
4: -74.2353668, 122.3866501, -59.0593567, 98.2843475, -172.5197144, 181.4459991

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_B2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5179793, upper bound: 86.5191001
time: 1.14 seconds

## Relational analysis of NS_B2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_B2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5178402, upper bound: 86.5191001
time: 0.71 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -35.5939522, 79.3213425, -26.6515274, 59.4921684, -95.0861206, 105.9728622
1: -72.5502701, 118.1737442, -54.8183479, 89.0429993, -161.5932465, 172.9920349
2: -55.9251175, 113.8886566, -42.1476860, 85.8687134, -141.7938232, 156.0363159
3: -85.1936264, 136.4088593, -64.1015701, 102.9113312, -188.1049500, 200.5104370
4: -78.5589142, 130.4744110, -59.0593567, 98.2843475, -176.8432617, 189.5337677

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_B2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5174557, upper bound: 86.5179209
time: 0.72 seconds

## Relational analysis of NS_B2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_B2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5168017, upper bound: 86.5181922
time: 0.72 seconds

## BFS NS instance: NS_B2_B1_A1_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -24.4952698, 54.2567024, -34.6734505, 76.7171021, -101.2123718, 88.9301300
1: -50.1875725, 81.0191422, -70.4962769, 114.1553574, -164.3429260, 151.5153961
2: -38.6676903, 78.2001114, -54.4049644, 110.1817398, -148.8494263, 132.6050720
3: -58.7932396, 93.5394440, -82.8818741, 131.7398529, -190.5330505, 176.4212952
4: -54.2706947, 89.4193649, -76.5031128, 126.0130692, -180.2837219, 165.9224701

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A1_B1

### Relational analysis result of NS_B2_B1_A1_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5189080, upper bound: 86.5179793
time: 0.65 seconds

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A1_B2

### Relational analysis result of NS_B2_B1_A1_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5189080, upper bound: 86.5184509
time: 0.69 seconds

## BFS NS instance: NS_B2_B1_A1_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -24.4805527, 54.1118813, -34.1773033, 75.4971313, -99.9776840, 88.2891846
1: -50.2658539, 80.6154861, -69.4185791, 112.2734604, -162.5393066, 150.0340424
2: -38.6476822, 77.8761292, -53.6015778, 108.3682404, -147.0159302, 131.4776764
3: -58.7916985, 93.0441208, -81.6559677, 129.5575562, -188.3492432, 174.7000885
4: -54.1482506, 89.0867920, -75.4094315, 123.9236069, -178.0718384, 164.4962158

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A2_A1

### Relational analysis result of NS_B2_B1_A1_B2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5164144, upper bound: 86.5160084
time: 0.71 seconds

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A2_A2

### Relational analysis result of NS_B2_B1_A1_B2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169680, upper bound: 86.5160730
time: 0.68 seconds

## BFS NS instance: NS_B2_B1_A1_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -26.3134155, 58.7478333, -34.5954552, 76.5185928, -102.8320084, 93.3432922
1: -54.0844612, 87.9377365, -70.2902069, 113.8501892, -167.9346313, 158.2279358
2: -41.6011887, 84.7856674, -54.2694664, 109.8795395, -151.4807281, 139.0551147
3: -63.2707901, 101.6368179, -82.6709671, 131.3857727, -194.6565094, 184.3077850
4: -58.3161469, 97.0519028, -76.3321686, 125.6706924, -183.9868469, 173.3840332

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A1_B2_A1_A2_A1_A1

### Relational analysis result of NS_B2_B1_A1_B2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5179209, upper bound: 86.5176775
time: 0.86 seconds

## Relational analysis of NS_B2_B1_A1_B2_A1_A2_A1_A2

### Relational analysis result of NS_B2_B1_A1_B2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5181922, upper bound: 86.5174941
time: 0.83 seconds

## BFS NS instance: NS_B2_B1_A1_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -25.7846470, 57.5922928, -34.1009903, 75.3037872, -101.0884323, 91.6932831
1: -53.2246819, 86.0141983, -69.2204208, 111.9737167, -165.1983948, 155.2346039
2: -40.8127785, 83.0171814, -53.4699745, 108.0745697, -148.8873444, 136.4871521
3: -62.0796204, 99.3825912, -81.4517517, 129.2108459, -191.2904663, 180.8343506
4: -57.0483360, 95.0974655, -75.2433167, 123.5904770, -180.6388092, 170.3407440

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A1_B2_A1_A2_A2_A1

### Relational analysis result of NS_B2_B1_A1_B2_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5176128, upper bound: 86.5175470
time: 0.95 seconds

## Relational analysis of NS_B2_B1_A1_B2_A1_A2_A2_A2

### Relational analysis result of NS_B2_B1_A1_B2_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5181922, upper bound: 86.5174941
time: 0.69 seconds

## BFS NS instance: NS_B2_B1_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -34.6841049, 76.7415085, -33.1891823, 73.5602341, -107.9220123, 109.6219940
1: -70.5182114, 114.1923141, -67.4174957, 109.3930054, -179.9112244, 181.6097717
2: -54.4218407, 110.2171631, -52.0470467, 105.4886703, -159.9105072, 162.2642059
3: -82.9073639, 131.7827759, -79.3054047, 126.1759644, -209.0833282, 211.0881500
4: -76.5263824, 126.0537949, -73.2262878, 120.7436371, -196.8367004, 198.8830872

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A1_B2_A2_B1_B1_A1

### Relational analysis result of NS_B2_B1_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5182092, upper bound: 86.5176963
time: 1.10 seconds

## Relational analysis of NS_B2_B1_A1_B2_A2_B1_B1_A2

### Relational analysis result of NS_B2_B1_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5182092, upper bound: 86.5176963
time: 0.93 seconds

## BFS NS instance: NS_B2_B1_A1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -34.1875801, 75.5207901, -33.3211670, 73.7641220, -107.6396713, 108.5272522
1: -69.4397964, 112.3093414, -67.8136978, 109.5202713, -178.9600677, 180.1230316
2: -53.6178665, 108.4025574, -52.2713203, 105.5861359, -159.2040100, 160.6738739
3: -81.6805954, 129.5991516, -79.6539383, 126.2583237, -207.9389191, 209.2530823
4: -75.4319077, 123.9631348, -73.4420395, 120.9325333, -195.9934082, 197.0630035

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A1_B2_A2_B1_B2_B1

### Relational analysis result of NS_B2_B1_A1_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5174720, upper bound: 86.5167412
time: 1.06 seconds

## Relational analysis of NS_B2_B1_A1_B2_A2_B1_B2_B2

### Relational analysis result of NS_B2_B1_A1_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5163668, upper bound: 86.5167818
time: 0.74 seconds

## BFS NS instance: NS_B2_B1_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -34.1801338, 75.6404953, -36.4004021, 81.5071945, -115.2779617, 111.7232513
1: -69.4390335, 112.5254822, -74.2978668, 121.7082825, -191.0514679, 186.8233490
2: -53.6195335, 108.5755005, -57.2228317, 117.0725632, -170.6920929, 165.7983398
3: -81.6737976, 129.8419952, -87.1980209, 140.4172974, -222.0148010, 217.0399780
4: -75.4179764, 124.2033615, -80.3306427, 134.2094879, -208.9923096, 204.1273041

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_B2_B1_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5159738, upper bound: 86.5161019
time: 0.71 seconds

## Relational analysis of NS_B2_B1_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_B2_B1_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5159738, upper bound: 86.5159951
time: 0.72 seconds

## BFS NS instance: NS_B2_B1_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -34.5999908, 76.5457153, -34.9840469, 78.0578156, -112.3249130, 111.2515717
1: -70.2979584, 113.8941956, -71.2602234, 116.2905121, -186.5884705, 185.1543884
2: -54.2753105, 109.9115982, -54.9437561, 112.0266495, -166.3019562, 164.8553467
3: -82.6829147, 131.4288025, -83.7196884, 134.2090759, -216.8919678, 215.1484985
4: -76.3385849, 125.7054825, -77.2100067, 128.3615112, -204.2482300, 202.6003876

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_B2_A2_B2_B2_A1

### Relational analysis result of NS_B2_B1_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151787, upper bound: 86.5151787
time: 0.75 seconds

## Relational analysis of NS_B2_B1_A1_B2_A2_B2_B2_A2

### Relational analysis result of NS_B2_B1_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5151787, upper bound: 86.5151787
time: 0.64 seconds

## BFS NS instance: NS_B2_B1_A2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -38.8404617, 85.7235718, -24.7947807, 54.9026337, -93.7430878, 110.5183487
1: -78.6360092, 127.8507690, -50.8208466, 81.9825592, -160.6185608, 178.6716156
2: -60.8239326, 123.0352936, -39.1448898, 79.1387329, -139.9626617, 162.1801758
3: -92.4954605, 147.4240570, -59.5194817, 94.6461639, -187.1416168, 206.9435425
4: -85.5664825, 141.0511780, -54.9264069, 90.4786148, -175.8664246, 195.9775848

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A2_B1_A1_A1_B1_A1

### Relational analysis result of NS_B2_B1_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5146923, upper bound: 86.5169524
time: 0.68 seconds

## Relational analysis of NS_B2_B1_A2_B1_A1_A1_B1_A2

### Relational analysis result of NS_B2_B1_A2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5141978, upper bound: 86.5167144
time: 1.05 seconds

## BFS NS instance: NS_B2_B1_A2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -38.8451805, 85.7352219, -26.6038094, 59.3871193, -98.2045822, 112.3390198
1: -78.6439362, 127.8678589, -54.7143936, 88.8858185, -167.5297546, 182.5822449
2: -60.8310814, 123.0488586, -42.0707512, 85.7126160, -146.5437012, 165.1196136
3: -92.5059967, 147.4409790, -63.9839859, 102.7267075, -195.2326813, 211.4249573
4: -85.5767593, 141.0693512, -58.9548454, 98.1071320, -183.4249115, 200.0241852

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A2_B1_A1_A1_B2_A1

### Relational analysis result of NS_B2_B1_A2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5146923, upper bound: 86.5175238
time: 0.70 seconds

## Relational analysis of NS_B2_B1_A2_B1_A1_A1_B2_A2

### Relational analysis result of NS_B2_B1_A2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5141978, upper bound: 86.5172735
time: 0.67 seconds

## BFS NS instance: NS_B2_B1_A2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -39.0223045, 86.1195297, -24.6229420, 54.5028534, -93.5251617, 110.7424698
1: -78.9967041, 128.5081787, -50.4295845, 81.3858566, -160.3825378, 178.9377594
2: -61.0987053, 123.6819763, -38.8616714, 78.5468369, -139.6455383, 162.5436401
3: -92.9216766, 148.2066956, -59.0867767, 93.9430542, -186.8646851, 207.2934723
4: -85.9620361, 141.7580109, -54.5465927, 89.8019714, -175.5879211, 196.3045807

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A2_B1_A1_A2_B1_A1

### Relational analysis result of NS_B2_B1_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5147394, upper bound: 86.5158108
time: 0.78 seconds

## Relational analysis of NS_B2_B1_A2_B1_A1_A2_B1_A2

### Relational analysis result of NS_B2_B1_A2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5142454, upper bound: 86.5152602
time: 1.13 seconds

## BFS NS instance: NS_B2_B1_A2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -39.0269356, 86.1308899, -26.4298992, 58.9823875, -97.9839478, 112.5607910
1: -79.0043106, 128.5249939, -54.3194427, 88.2814178, -167.2857056, 182.8443756
2: -61.1056557, 123.6950684, -41.7842941, 85.1107941, -146.2164459, 165.4793091
3: -92.9319534, 148.2230682, -63.5466423, 102.0144424, -194.9463806, 211.7697144
4: -85.9720993, 141.7758026, -58.5708694, 97.4208832, -183.1371765, 200.3466797

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A2_B1_A1_A2_B2_A1

### Relational analysis result of NS_B2_B1_A2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5147394, upper bound: 86.5163998
time: 0.72 seconds

## Relational analysis of NS_B2_B1_A2_B1_A1_A2_B2_A2

### Relational analysis result of NS_B2_B1_A2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5142454, upper bound: 86.5158265
time: 0.77 seconds

## BFS NS instance: NS_B2_B1_A2_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -40.2270508, 89.2425156, -25.6544609, 56.6676674, -96.8590393, 114.8969650
1: -81.6104202, 133.2597504, -52.5431938, 84.6382980, -166.2487030, 185.8029480
2: -63.0631714, 128.1449738, -40.4881821, 81.7585983, -144.8217773, 168.6331482
3: -95.8974991, 153.7331543, -61.5707207, 97.7388382, -193.6363373, 215.3038788
4: -88.6445770, 147.0542297, -56.8330002, 93.4179459, -181.8170471, 203.8872223

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B2_B1_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A2_B1_A2_A1_A1_B1

### Relational analysis result of NS_B2_B1_A2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157088, upper bound: 86.5172038
time: 0.77 seconds

## Relational analysis of NS_B2_B1_A2_B1_A2_A1_A1_B2

### Relational analysis result of NS_B2_B1_A2_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157088, upper bound: 86.5175238
time: 0.80 seconds

## BFS NS instance: NS_B2_B1_A2_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -39.8324852, 88.3369827, -25.1059895, 55.3168106, -95.1427612, 113.4429703
1: -81.0726166, 131.7121735, -51.3711090, 82.5406265, -163.6132355, 183.0832825
2: -62.4903297, 126.7340775, -39.6024590, 79.7721252, -142.2624512, 166.3365326
3: -95.0561829, 151.9526520, -60.2270126, 95.3214645, -190.3776550, 212.1796570
4: -87.6747589, 145.4871216, -55.6243134, 91.1271133, -178.7177124, 201.1114349

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A2_B1_A2_A1_A2_B1

### Relational analysis result of NS_B2_B1_A2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5154486, upper bound: 86.5172735
time: 1.57 seconds

## Relational analysis of NS_B2_B1_A2_B1_A2_A1_A2_B2

### Relational analysis result of NS_B2_B1_A2_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5154486, upper bound: 86.5172735
time: 1.07 seconds

## BFS NS instance: NS_B2_B1_A2_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -40.6718597, 90.1731949, -25.4846401, 56.2726021, -96.9102707, 115.6578369
1: -82.5060959, 134.6882629, -52.1578369, 84.0474396, -166.5535278, 186.8460846
2: -63.7547302, 129.5319672, -40.2086258, 81.1725922, -144.9273224, 169.7405853
3: -96.9525223, 155.3708191, -61.1443405, 97.0434418, -193.9959412, 216.5151672
4: -89.6093292, 148.6272430, -56.4580841, 92.7506104, -182.1214294, 205.0853271

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B2_B1_A2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A2_B1_A2_A2_A1_B1

### Relational analysis result of NS_B2_B1_A2_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157954, upper bound: 86.5167758
time: 1.13 seconds

## Relational analysis of NS_B2_B1_A2_B1_A2_A2_A1_B2

### Relational analysis result of NS_B2_B1_A2_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157954, upper bound: 86.5167758
time: 0.70 seconds

## BFS NS instance: NS_B2_B1_A2_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -40.0497055, 88.7359390, -24.9353600, 54.9206619, -94.9663467, 113.6712952
1: -81.4523087, 132.3410645, -50.9852943, 81.9476852, -163.3999939, 183.3263245
2: -62.8121147, 127.3428802, -39.3218803, 79.1847610, -141.9968719, 166.6647644
3: -95.5325546, 152.6627655, -59.7993507, 94.6261749, -190.1587219, 212.4621124
4: -88.1388779, 146.1805420, -55.2478218, 90.4582291, -178.5197601, 201.4283600

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A2_B1_A2_A2_A2_B1

### Relational analysis result of NS_B2_B1_A2_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157879, upper bound: 86.5170950
time: 0.72 seconds

## Relational analysis of NS_B2_B1_A2_B1_A2_A2_A2_B2

### Relational analysis result of NS_B2_B1_A2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157879, upper bound: 86.5170950
time: 0.66 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -39.3379364, 87.4714813, -34.1688118, 75.6383057, -114.4871063, 121.2294693
1: -79.9524155, 130.6604156, -69.4431305, 112.5318832, -192.4842834, 200.0060425
2: -61.7286949, 125.6976700, -53.6075134, 108.5771408, -170.3058319, 179.3051453
3: -93.8082657, 150.7458038, -81.6606903, 129.8402100, -223.6368713, 232.4064789
4: -86.6656494, 144.1406708, -75.3911514, 124.2047119, -210.0189514, 218.9138336

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_B2_A1_A1_A1_B1

### Relational analysis result of NS_B2_B1_A2_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5123203, upper bound: 86.5147120
time: 0.73 seconds

## Relational analysis of NS_B2_B1_A2_B2_A1_A1_A1_B2

### Relational analysis result of NS_B2_B1_A2_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5123203, upper bound: 86.5147120
time: 0.68 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -39.7989845, 88.0610199, -33.4485626, 74.1411896, -113.4711609, 121.1772385
1: -80.9523926, 131.2895203, -67.9643478, 110.2673035, -191.2196960, 199.2538605
2: -62.4482765, 126.6562119, -52.4615517, 106.3287354, -168.7770081, 179.1177673
3: -94.9367447, 151.6725006, -79.9348450, 127.1763992, -222.1131439, 231.6073151
4: -87.6485443, 145.0855255, -73.7933884, 121.6912766, -208.5423584, 218.4455566

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_B1_A2_B2_A1_A2_B1_B1

### Relational analysis result of NS_B2_B1_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5135943, upper bound: 86.5144031
time: 0.79 seconds

## Relational analysis of NS_B2_B1_A2_B2_A1_A2_B1_B2

### Relational analysis result of NS_B2_B1_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5135943, upper bound: 86.5144031
time: 1.02 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -39.8075905, 88.0818481, -35.3848000, 78.8857727, -118.1875229, 123.1420517
1: -80.9696655, 131.3217316, -72.1074982, 117.5246582, -198.4943085, 203.4292297
2: -62.4617386, 126.6856003, -55.5884247, 113.2461090, -175.7078400, 182.2740173
3: -94.9574356, 151.7088470, -84.6876907, 135.6499786, -230.5912933, 236.3965454
4: -87.6673126, 145.1202545, -78.0964890, 129.7453766, -216.5247955, 222.7978210

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_B1_A2_B2_A1_A2_B2_B1

### Relational analysis result of NS_B2_B1_A2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5135943, upper bound: 86.5144031
time: 1.23 seconds

## Relational analysis of NS_B2_B1_A2_B2_A1_A2_B2_B2

### Relational analysis result of NS_B2_B1_A2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5135943, upper bound: 86.5144031
time: 0.73 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -41.2276497, 91.5736542, -33.5948067, 74.4358368, -115.1932068, 124.7996063
1: -83.9667435, 136.6479492, -68.2697296, 110.7057419, -194.6724854, 204.9176788
2: -64.7444916, 131.6818237, -52.6958694, 106.7617035, -171.5061951, 184.3776855
3: -98.4125214, 157.8638153, -80.2855911, 127.6881714, -226.1006927, 238.1494141
4: -90.8285828, 151.0263214, -74.1171494, 122.1819382, -212.2085876, 224.6219940

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A2_B2_A2_B1_A1_A1

### Relational analysis result of NS_B2_B1_A2_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5146718, upper bound: 86.5167089
time: 0.77 seconds

## Relational analysis of NS_B2_B1_A2_B2_A2_B1_A1_A2

### Relational analysis result of NS_B2_B1_A2_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5144497, upper bound: 86.5164779
time: 0.65 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -41.5474167, 92.2316589, -33.3906860, 73.9694672, -115.0443649, 125.2644119
1: -84.5793991, 137.6732330, -67.8098831, 110.0113373, -194.5907288, 205.4831238
2: -65.2294846, 132.6773224, -52.3601761, 106.0747986, -171.3042908, 185.0375061
3: -99.1497269, 159.0365295, -79.7755966, 126.8695221, -226.0192566, 238.8121338
4: -91.5153961, 152.1526794, -73.6662216, 121.3947754, -212.1076965, 225.3389435

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A2_B2_A2_B1_A2_A1

### Relational analysis result of NS_B2_B1_A2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5147865, upper bound: 86.5162957
time: 1.04 seconds

## Relational analysis of NS_B2_B1_A2_B2_A2_B1_A2_A2

### Relational analysis result of NS_B2_B1_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5147850, upper bound: 86.5162956
time: 0.66 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -41.2363663, 91.5947342, -35.5375557, 79.1955261, -119.9314651, 126.7784500
1: -83.9842300, 136.6804810, -72.4283600, 117.9850235, -201.9692383, 209.1088257
2: -64.7581177, 131.7114868, -55.8339577, 113.7026901, -178.4608002, 187.5454407
3: -98.4334412, 157.9005432, -85.0555267, 136.1874542, -234.6208954, 242.9560699
4: -90.8475723, 151.0613556, -78.4352036, 130.2622833, -220.2363434, 229.0102234

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A2_B2_A2_B2_A1_A1

### Relational analysis result of NS_B2_B1_A2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5154731, upper bound: 86.5170536
time: 0.89 seconds

## Relational analysis of NS_B2_B1_A2_B2_A2_B2_A1_A2

### Relational analysis result of NS_B2_B1_A2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5154081, upper bound: 86.5169812
time: 0.75 seconds

## BFS NS instance: NS_B2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -41.5563622, 92.2530746, -35.3571854, 78.7792206, -119.8338013, 127.2676620
1: -84.5971527, 137.7064056, -72.0172348, 117.3651886, -201.9623413, 209.7236176
2: -65.2434311, 132.7070770, -55.5360146, 113.0856171, -178.3290253, 188.2430878
3: -99.1711349, 159.0738525, -84.6022568, 135.4549103, -234.6260376, 243.6761017
4: -91.5348587, 152.1878204, -78.0367279, 129.5575104, -220.2198181, 229.7806396

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A2_B2_A2_B2_A2_A1

### Relational analysis result of NS_B2_B1_A2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157306, upper bound: 86.5168705
time: 0.65 seconds

## Relational analysis of NS_B2_B1_A2_B2_A2_B2_A2_A2

### Relational analysis result of NS_B2_B1_A2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5157306, upper bound: 86.5168705
time: 0.76 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -27.1373177, 59.9452400, -41.8306122, 92.8024368, -119.9397430, 101.7317200
1: -56.0290947, 89.5211945, -85.2682266, 138.6649475, -194.6940308, 174.7894135
2: -42.9724541, 86.9908295, -65.7327728, 133.6516266, -176.6240845, 152.7235718
3: -65.2834015, 103.8596878, -99.8989105, 160.1092834, -225.3926697, 203.7585907
4: -60.0446091, 99.1446838, -92.1472244, 153.1213074, -213.1659241, 191.0245514

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_A1_B1_B1_A1_B1

### Relational analysis result of NS_B2_B2_A1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5164346, upper bound: 86.5152710
time: 0.71 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_B1_A1_B2

### Relational analysis result of NS_B2_B2_A1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5164346, upper bound: 86.5152710
time: 0.73 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -26.6787815, 58.9494514, -41.2498245, 91.6173096, -118.2960739, 100.1985245
1: -54.9567108, 88.0640106, -84.0043259, 136.9056396, -191.8623352, 172.0682983
2: -42.1733932, 85.5224380, -64.7819595, 131.8560791, -174.0294647, 150.3043976
3: -64.1343613, 102.1489258, -98.4780579, 158.0316010, -222.1659546, 200.6269226
4: -58.9961700, 97.5008698, -90.8636780, 151.1203156, -210.1164551, 188.1720886

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B2_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_A1_B1_B1_A2_B1

### Relational analysis result of NS_B2_B2_A1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5164346, upper bound: 86.5152710
time: 0.72 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_B1_A2_B2

### Relational analysis result of NS_B2_B2_A1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5164346, upper bound: 86.5152710
time: 0.84 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -29.6789055, 65.8599548, -41.8232574, 92.3237762, -122.0026703, 107.6073914
1: -61.4777107, 98.5587463, -85.1723709, 137.7249756, -199.2026825, 183.7311096
2: -47.0591927, 95.7345810, -65.6709366, 132.9873657, -180.0465546, 161.4055176
3: -71.5136795, 114.2331772, -99.8372116, 159.1418915, -230.6555481, 214.0447998
4: -65.6465988, 109.1092682, -92.1145096, 152.2563782, -217.9029846, 200.8768768

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B2_A1_A1_B1_B2_A1_B1

### Relational analysis result of NS_B2_B2_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5146664, upper bound: 86.5121202
time: 0.79 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_A1_B1_B2_A1_A1

### Relational analysis result of NS_B2_B2_A1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5161085, upper bound: 86.5149771
time: 0.70 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_B2_A1_A1_B1_B2_A1_B1

### Relational analysis result of NS_B2_B2_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5162957, upper bound: 86.5151372
time: 0.80 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_B2_A1_B2

### Relational analysis result of NS_B2_B2_A1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5160873, upper bound: 86.5149516
time: 0.68 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -28.1758842, 62.1949158, -41.8073273, 92.2852783, -120.4611511, 103.9980240
1: -58.1934853, 92.8914032, -85.1428299, 137.6663666, -195.8598480, 178.0341797
2: -44.6200180, 90.3255768, -65.6462860, 132.9369812, -177.5570068, 155.9718628
3: -67.8013153, 107.7304535, -99.8000946, 159.0789032, -226.8802185, 207.5305481
4: -62.3211555, 102.9019775, -92.0798721, 152.1939392, -214.5150909, 194.8331146

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_A1_B1_B2_A2_A1

### Relational analysis result of NS_B2_B2_A1_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5161085, upper bound: 86.5149873
time: 0.69 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B2_A1_A1_B1_B2_A2_B1

### Relational analysis result of NS_B2_B2_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5146664, upper bound: 86.5122512
time: 0.74 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_B2_A1_A1_B1_B2_A2_B1

### Relational analysis result of NS_B2_B2_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5162957, upper bound: 86.5151417
time: 0.74 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_B2_A2_B2

### Relational analysis result of NS_B2_B2_A1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5160873, upper bound: 86.5149649
time: 0.76 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -28.0322666, 61.8337288, -44.9479294, 99.4538193, -127.4860840, 106.7816315
1: -57.8381958, 92.3545532, -91.8263245, 148.6888275, -206.5270233, 184.1808624
2: -44.3756371, 89.7799149, -70.6630096, 143.6418457, -188.0174866, 160.4429321
3: -67.4191055, 107.0806427, -107.4487076, 171.8326569, -239.2517700, 214.5293427
4: -62.0065613, 102.2750854, -99.0214844, 164.4249420, -226.4315033, 201.1890717

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A1_A1_B2_A1_A1_B1

### Relational analysis result of NS_B2_B2_A1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143369, upper bound: 86.5128703
time: 0.77 seconds

## Relational analysis of NS_B2_B2_A1_A1_B2_A1_A1_B2

### Relational analysis result of NS_B2_B2_A1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5143369, upper bound: 86.5128703
time: 0.71 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -42.3381729, 93.5244446, -40.1815834, 89.0718384, -130.7341003, 133.1429596
1: -86.2220154, 139.5653534, -81.5323486, 133.1817780, -219.1932068, 221.0963440
2: -66.4798203, 134.7054291, -62.9949646, 128.1665192, -194.4541168, 197.7003784
3: -101.0780792, 161.2415619, -95.7744293, 153.6057587, -254.2700195, 256.9435730
4: -93.2408371, 154.2395782, -88.4871368, 146.9062653, -238.8093262, 241.6976013

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A2_A2_B1_A2_B1_B1

### Relational analysis result of NS_B2_B2_A2_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5120714, upper bound: 86.5115089
time: 0.69 seconds

## Relational analysis of NS_B2_B2_A2_A2_B1_A2_B1_B2

### Relational analysis result of NS_B2_B2_A2_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5048293, upper bound: 86.5044812
time: 0.77 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -42.3381729, 93.5244446, -42.8787460, 95.2026138, -136.9158478, 135.8546295
1: -86.2220154, 139.5653534, -87.4665222, 142.3339233, -228.4445343, 227.0318756
2: -66.4798203, 134.7054291, -67.3983231, 137.1517639, -203.5497742, 202.1037598
3: -101.0780792, 161.2415619, -102.4452057, 164.3228455, -265.1225281, 263.6191711
4: -93.2408371, 154.2395782, -94.4505920, 157.1551514, -249.1661377, 247.6482086

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A2_A2_B1_A2_B2_B1

### Relational analysis result of NS_B2_B2_A2_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5120714, upper bound: 86.5122913
time: 0.73 seconds

## Relational analysis of NS_B2_B2_A2_A2_B1_A2_B2_B2

### Relational analysis result of NS_B2_B2_A2_A2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5048293, upper bound: 86.5076691
time: 1.09 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -42.3381729, 93.5244446, -42.3273506, 93.4989471, -135.3140411, 135.3288727
1: -86.2220154, 139.5653534, -86.1991959, 139.5266571, -225.7486572, 225.7645111
2: -66.4798203, 134.7054291, -66.4623184, 134.6683197, -201.1481323, 201.1677399
3: -101.0780792, 161.2415619, -101.0516281, 161.1966248, -262.2703857, 262.2895203
4: -93.2408371, 154.2395782, -93.2168884, 154.1968231, -246.4683685, 246.4876099

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_B2_B1

### Relational analysis result of NS_B2_B2_A2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5129908, upper bound: 86.5141183
time: 0.78 seconds

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_B2_B2

### Relational analysis result of NS_B2_B2_A2_A2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -86.5133357, upper bound: 86.5140279
time: 0.72 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.46 seconds
NS_B1_B2_A2_A2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5103501, upper bound: 86.5145621
NS_B1_B2_A2_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5103501, upper bound: 86.5150166
NS_B2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5181982, upper bound: 86.5186575
NS_B2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5181982, upper bound: 86.5188480
NS_B2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5188585, upper bound: 86.5186602
NS_B2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5188585, upper bound: 86.5188505
NS_B2_B1_A1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5181932, upper bound: 86.5192403
NS_B2_B1_A1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5184326, upper bound: 86.5188585
NS_B2_B1_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5181932, upper bound: 86.5192925
NS_B2_B1_A1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5184326, upper bound: 86.5184326
NS_B2_B1_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5179793, upper bound: 86.5189080
NS_B2_B1_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5179793, upper bound: 86.5189080
NS_B2_B1_A1_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5160084, upper bound: 86.5164144
NS_B2_B1_A1_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5160730, upper bound: 86.5169680
NS_B2_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5179793, upper bound: 86.5191001
NS_B2_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5178402, upper bound: 86.5191001
NS_B2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5174557, upper bound: 86.5179209
NS_B2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5168017, upper bound: 86.5181922
NS_B2_B1_A1_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5189080, upper bound: 86.5179793
NS_B2_B1_A1_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5189080, upper bound: 86.5184509
NS_B2_B1_A1_B2_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5164144, upper bound: 86.5160084
NS_B2_B1_A1_B2_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5169680, upper bound: 86.5160730
NS_B2_B1_A1_B2_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5179209, upper bound: 86.5176775
NS_B2_B1_A1_B2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5181922, upper bound: 86.5174941
NS_B2_B1_A1_B2_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5176128, upper bound: 86.5175470
NS_B2_B1_A1_B2_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5181922, upper bound: 86.5174941
NS_B2_B1_A1_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5182092, upper bound: 86.5176963
NS_B2_B1_A1_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5182092, upper bound: 86.5176963
NS_B2_B1_A1_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5174720, upper bound: 86.5167412
NS_B2_B1_A1_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5163668, upper bound: 86.5167818
NS_B2_B1_A1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5159738, upper bound: 86.5161019
NS_B2_B1_A1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5159738, upper bound: 86.5159951
NS_B2_B1_A1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5151787, upper bound: 86.5151787
NS_B2_B1_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5151787, upper bound: 86.5151787
NS_B2_B1_A2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5146923, upper bound: 86.5169524
NS_B2_B1_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5141978, upper bound: 86.5167144
NS_B2_B1_A2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5146923, upper bound: 86.5175238
NS_B2_B1_A2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5141978, upper bound: 86.5172735
NS_B2_B1_A2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5147394, upper bound: 86.5158108
NS_B2_B1_A2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5142454, upper bound: 86.5152602
NS_B2_B1_A2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5147394, upper bound: 86.5163998
NS_B2_B1_A2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5142454, upper bound: 86.5158265
NS_B2_B1_A2_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5157088, upper bound: 86.5172038
NS_B2_B1_A2_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5157088, upper bound: 86.5175238
NS_B2_B1_A2_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5154486, upper bound: 86.5172735
NS_B2_B1_A2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5154486, upper bound: 86.5172735
NS_B2_B1_A2_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5157954, upper bound: 86.5167758
NS_B2_B1_A2_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5157954, upper bound: 86.5167758
NS_B2_B1_A2_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5157879, upper bound: 86.5170950
NS_B2_B1_A2_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5157879, upper bound: 86.5170950
NS_B2_B1_A2_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5123203, upper bound: 86.5147120
NS_B2_B1_A2_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5123203, upper bound: 86.5147120
NS_B2_B1_A2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5135943, upper bound: 86.5144031
NS_B2_B1_A2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5135943, upper bound: 86.5144031
NS_B2_B1_A2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5135943, upper bound: 86.5144031
NS_B2_B1_A2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5135943, upper bound: 86.5144031
NS_B2_B1_A2_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5146718, upper bound: 86.5167089
NS_B2_B1_A2_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5144497, upper bound: 86.5164779
NS_B2_B1_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5147865, upper bound: 86.5162957
NS_B2_B1_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5147850, upper bound: 86.5162956
NS_B2_B1_A2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5154731, upper bound: 86.5170536
NS_B2_B1_A2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5154081, upper bound: 86.5169812
NS_B2_B1_A2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5157306, upper bound: 86.5168705
NS_B2_B1_A2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5157306, upper bound: 86.5168705
NS_B2_B2_A1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5164346, upper bound: 86.5152710
NS_B2_B2_A1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5164346, upper bound: 86.5152710
NS_B2_B2_A1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5164346, upper bound: 86.5152710
NS_B2_B2_A1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5164346, upper bound: 86.5152710
NS_B2_B2_A1_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5162957, upper bound: 86.5151372
NS_B2_B2_A1_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5160873, upper bound: 86.5149516
NS_B2_B2_A1_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5162957, upper bound: 86.5151417
NS_B2_B2_A1_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5160873, upper bound: 86.5149649
NS_B2_B2_A1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5143369, upper bound: 86.5128703
NS_B2_B2_A1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5143369, upper bound: 86.5128703
NS_B2_B2_A2_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5120714, upper bound: 86.5115089
NS_B2_B2_A2_A2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5048293, upper bound: 86.5044812
NS_B2_B2_A2_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5120714, upper bound: 86.5122913
NS_B2_B2_A2_A2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5048293, upper bound: 86.5076691
NS_B2_B2_A2_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5129908, upper bound: 86.5141183
NS_B2_B2_A2_A2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 8, time: 2.46
Output dim: 0, lower bound: -86.5133357, upper bound: 86.5140279

## BFS NS instance: NS_B1_B2_A2_A2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -44.7875900, 97.9924545, -37.5218811, 81.4701233, -125.9659500, 135.0523529
1: -91.0842438, 146.2351074, -76.3807297, 121.2729340, -212.3571472, 222.6158447
2: -70.2225037, 141.3792877, -58.8283234, 118.3260880, -188.5485687, 200.2076111
3: -106.9633636, 168.9033508, -89.4478455, 141.0436249, -248.0069733, 258.3511963
4: -98.6951752, 161.6355286, -82.6951370, 134.3515472, -232.5333710, 243.5520325

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 20

## BFS NS instance: NS_B1_B2_A2_A2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -45.3600998, 99.6858749, -37.5218811, 81.4701233, -126.5311127, 136.8150787
1: -92.3488464, 148.7944489, -76.3807297, 121.2729340, -213.6217346, 225.1751709
2: -71.1861877, 143.7024231, -58.8283234, 118.3260880, -189.5122681, 202.5307465
3: -108.3705826, 171.8639069, -89.4478455, 141.0436249, -249.4141846, 261.3117676
4: -99.9259949, 164.4654083, -82.6951370, 134.3515472, -233.7157440, 246.5570526

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

## BFS NS instance: NS_B2_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -24.9929028, 55.1936455, -24.0152893, 53.1521416, -78.1450424, 79.2089386
1: -51.1022110, 82.4255676, -49.1030235, 79.3563385, -130.4585419, 131.5285950
2: -39.4201202, 79.5396347, -37.8782158, 76.5193558, -115.9394684, 117.4178467
3: -59.9365768, 95.1197815, -57.5911865, 91.5523987, -151.4889374, 152.7109680
4: -55.3781166, 90.9076996, -53.2148018, 87.5109482, -142.8890381, 144.1224976

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A1_B1_A1_B1_A1_B1_B1

### Relational analysis result of NS_B2_B1_A1_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5181805, upper bound: 86.5186320
time: 0.68 seconds

## Relational analysis of NS_B2_B1_A1_B1_A1_B1_A1_B1_B2

### Relational analysis result of NS_B2_B1_A1_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5180435, upper bound: 86.5186320
time: 0.76 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -24.9929028, 55.1936455, -24.0849476, 53.3482246, -78.3411255, 79.2785950
1: -51.1022110, 82.4255676, -49.2209282, 79.6906586, -130.7928772, 131.6464691
2: -39.4201202, 79.5396347, -37.9757004, 76.8186111, -116.2387314, 117.5153351
3: -59.9365768, 95.1197815, -57.7439003, 91.9270935, -151.8636780, 152.8636780
4: -55.3781166, 90.9076996, -53.3599625, 87.8598328, -143.2379456, 144.2676697

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A1_B1_A1_B1_A1_B2_B1

### Relational analysis result of NS_B2_B1_A1_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5181805, upper bound: 86.5188206
time: 0.94 seconds

## Relational analysis of NS_B2_B1_A1_B1_A1_B1_A1_B2_B2

### Relational analysis result of NS_B2_B1_A1_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5180435, upper bound: 86.5188189
time: 0.70 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -25.1023197, 55.4789200, -24.0152893, 53.1521416, -78.2544556, 79.4942093
1: -51.3078156, 82.8974152, -49.1030235, 79.3563385, -130.6641388, 132.0004272
2: -39.5814896, 79.9709396, -37.8782158, 76.5193558, -116.1008377, 117.8491516
3: -60.1899796, 95.6519241, -57.5911865, 91.5523987, -151.7423401, 153.2431030
4: -55.6106911, 91.4047470, -53.2148018, 87.5109482, -143.1216278, 144.6195374

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A1_B1_A1_B1_A2_B1_B1

### Relational analysis result of NS_B2_B1_A1_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5188585, upper bound: 86.5186602
time: 0.74 seconds

## Relational analysis of NS_B2_B1_A1_B1_A1_B1_A2_B1_B2

### Relational analysis result of NS_B2_B1_A1_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5186861, upper bound: 86.5186602
time: 0.70 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -25.1023197, 55.4789200, -24.0849476, 53.3482246, -78.4505234, 79.5638657
1: -51.3078156, 82.8974152, -49.2209282, 79.6906586, -130.9984741, 132.1183319
2: -39.5814896, 79.9709396, -37.9757004, 76.8186111, -116.4001007, 117.9466400
3: -60.1899796, 95.6519241, -57.7439003, 91.9270935, -152.1170654, 153.3958282
4: -55.6106911, 91.4047470, -53.3599625, 87.8598328, -143.4705200, 144.7647095

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_B2_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5155905, upper bound: 86.5160115
time: 0.68 seconds

## Relational analysis of NS_B2_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_B2_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5188585, upper bound: 86.5187796
time: 0.70 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -24.0263290, 53.1777229, -26.6038094, 59.3871193, -83.4134521, 79.7815247
1: -49.1259842, 79.3953247, -54.7143936, 88.8858185, -138.0117798, 134.1097107
2: -37.8957558, 76.5566483, -42.0707512, 85.7126160, -123.6083679, 118.6273956
3: -57.6178055, 91.5977478, -63.9839859, 102.7267075, -160.3444366, 155.5817108
4: -53.2389565, 87.5537491, -58.9548454, 98.1071320, -151.3460541, 146.5085602

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A1_B1_A1_B2_A1_A1_B1

### Relational analysis result of NS_B2_B1_A1_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5186575, upper bound: 86.5181982
time: 1.30 seconds

## Relational analysis of NS_B2_B1_A1_B1_A1_B2_A1_A1_B2

### Relational analysis result of NS_B2_B1_A1_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5186575, upper bound: 86.5188585
time: 0.83 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -24.0960484, 53.3737411, -26.4298992, 58.9823875, -83.0784378, 79.8036346
1: -49.2437172, 79.7294312, -54.3194427, 88.2814178, -137.5251312, 134.0488739
2: -37.9932442, 76.8556290, -41.7842941, 85.1107941, -123.1040344, 118.6399231
3: -57.7704697, 91.9721909, -63.5466423, 102.0144424, -159.7848816, 155.5188293
4: -53.3843040, 87.9024048, -58.5708694, 97.4208832, -150.8051758, 146.4732666

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A1_B1_A1_B2_A1_A2_B1

### Relational analysis result of NS_B2_B1_A1_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5188480, upper bound: 86.5181982
time: 1.02 seconds

## Relational analysis of NS_B2_B1_A1_B1_A1_B2_A1_A2_B2

### Relational analysis result of NS_B2_B1_A1_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5188480, upper bound: 86.5188585
time: 0.67 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -25.6854687, 57.3587494, -26.6038094, 59.3871193, -85.0725861, 83.9625473
1: -52.7105713, 85.8530426, -54.7143936, 88.8858185, -141.5963898, 140.5674286
2: -40.5870438, 82.6965561, -42.0707512, 85.7126160, -126.2996597, 124.7673035
3: -61.7211342, 99.1626663, -63.9839859, 102.7267075, -164.4478149, 163.1466522
4: -56.9416237, 94.6842957, -58.9548454, 98.1071320, -155.0487366, 153.6391296

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A1_B1_A1_B2_A2_A1_A1

### Relational analysis result of NS_B2_B1_A1_B1_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5184453, upper bound: 86.5192925
time: 0.76 seconds

## Relational analysis of NS_B2_B1_A1_B1_A1_B2_A2_A1_A2

### Relational analysis result of NS_B2_B1_A1_B1_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5184380, upper bound: 86.5192566
time: 1.26 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -26.0120983, 58.0795517, -26.4298992, 58.9823875, -84.9944839, 84.5094528
1: -53.3554573, 86.9672546, -54.3194427, 88.2814178, -141.6368561, 141.2866821
2: -41.0999146, 83.7591400, -41.7842941, 85.1107941, -126.2107010, 125.5434265
3: -62.4914093, 100.4365540, -63.5466423, 102.0144424, -164.5058289, 163.9832001
4: -57.6430588, 95.8966064, -58.5708694, 97.4208832, -155.0639343, 154.4674683

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A1_B1_A1_B2_A2_A2_B1

### Relational analysis result of NS_B2_B1_A1_B1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5192745, upper bound: 86.5186340
time: 0.76 seconds

## Relational analysis of NS_B2_B1_A1_B1_A1_B2_A2_A2_B2

### Relational analysis result of NS_B2_B1_A1_B1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5192745, upper bound: 86.5192852
time: 0.75 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -33.4623833, 74.1529846, -24.4840832, 54.2308731, -87.6932526, 98.6370697
1: -67.9932327, 110.2823792, -50.1643486, 80.9797668, -148.9729919, 160.4467316
2: -52.4842606, 106.3516541, -38.6499252, 78.1625061, -130.6467590, 145.0015717
3: -79.9663849, 127.2025146, -58.7663002, 93.4936752, -173.4600525, 185.9687653
4: -73.8259888, 121.7200012, -54.2462311, 89.3761597, -163.2021027, 175.9662323

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A1_B1_A2_B1_B1_A1_A1

### Relational analysis result of NS_B2_B1_A1_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5179793, upper bound: 86.5189080
time: 1.18 seconds

## Relational analysis of NS_B2_B1_A1_B1_A2_B1_B1_A1_A2

### Relational analysis result of NS_B2_B1_A1_B1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5179793, upper bound: 86.5189080
time: 0.75 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -35.4084320, 78.9219513, -24.4840832, 54.2308731, -89.6393051, 103.4060364
1: -72.1567764, 117.5756607, -50.1643486, 80.9797668, -153.1365356, 167.7400055
2: -55.6274071, 113.3097534, -38.6499252, 78.1625061, -133.7898865, 151.9596710
3: -84.7435913, 135.7206726, -58.7663002, 93.4936752, -178.2372437, 194.4869690
4: -78.1513367, 129.8157043, -54.2462311, 89.3761597, -167.5274811, 184.0619354

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A1_B1_A2_B1_B1_A2_A1

### Relational analysis result of NS_B2_B1_A1_B1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5179793, upper bound: 86.5189080
time: 1.02 seconds

## Relational analysis of NS_B2_B1_A1_B1_A2_B1_B1_A2_A2

### Relational analysis result of NS_B2_B1_A1_B1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5179793, upper bound: 86.5189080
time: 0.96 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -33.5372009, 74.1316299, -25.2139301, 56.1919365, -89.7291412, 99.3455582
1: -68.0829391, 110.2211914, -51.9403496, 84.0307846, -152.1136932, 162.1615448
2: -52.5886345, 106.3479233, -39.8626823, 80.8972855, -133.4859161, 146.2106018
3: -80.1076965, 127.1652756, -60.6574402, 96.8882904, -176.9959564, 187.8227234
4: -74.0007248, 121.6473999, -55.7739677, 92.6380157, -166.6386414, 177.4213715

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_B1_A2_B1_B2_B1_A1

### Relational analysis result of NS_B2_B1_A1_B1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5160084, upper bound: 86.5164144
time: 0.93 seconds

## Relational analysis of NS_B2_B1_A1_B1_A2_B1_B2_B1_A2

### Relational analysis result of NS_B2_B1_A1_B1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5160084, upper bound: 86.5164144
time: 0.59 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -33.9999046, 75.1324844, -23.9287663, 52.9399185, -86.9398193, 99.0612411
1: -69.0418167, 111.7307587, -49.0661011, 78.8630676, -147.9048767, 160.7968597
2: -53.3157616, 107.8289261, -37.7558174, 76.1276245, -129.4433746, 145.5847321
3: -81.2266769, 128.9217834, -57.4387054, 90.9825897, -172.2092438, 186.3604889
4: -75.0166397, 123.3127060, -52.9333153, 87.1144333, -162.1310577, 176.2460175

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A1_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B2_B1_A1_B1_A2_B1_B2_B2_A1

### Relational analysis result of NS_B2_B1_A1_B1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5160730, upper bound: 86.5169680
time: 0.76 seconds

## Relational analysis of NS_B2_B1_A1_B1_A2_B1_B2_B2_A2

### Relational analysis result of NS_B2_B1_A1_B1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5160730, upper bound: 86.5169680
time: 1.05 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -33.4623833, 74.1529846, -26.3029232, 58.7233849, -92.1857681, 100.4559097
1: -67.9932327, 110.2823792, -54.0627670, 87.9004898, -155.8937225, 164.3451538
2: -52.4842606, 106.3516541, -41.5845184, 84.7501526, -137.2344055, 147.9361725
3: -79.9663849, 127.2025146, -63.2455330, 101.5935440, -181.5599365, 190.4480133
4: -73.8259888, 121.7200012, -58.2931786, 97.0110016, -170.8369751, 180.0131836

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A1_B1_A2_B2_A1_B1_B1

### Relational analysis result of NS_B2_B1_A1_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5170979, upper bound: 86.5169155
time: 0.95 seconds

## Relational analysis of NS_B2_B1_A1_B1_A2_B2_A1_B1_B2

### Relational analysis result of NS_B2_B1_A1_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169114, upper bound: 86.5171430
time: 0.69 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -32.9635925, 72.9293594, -25.7729664, 57.5658112, -90.5294037, 98.7023239
1: -66.9095840, 108.3952942, -53.2008629, 85.9740067, -152.8835907, 161.5961609
2: -51.6769485, 104.5334244, -40.7943153, 82.9786682, -134.6556091, 145.3277435
3: -78.7336197, 125.0120087, -62.0517235, 99.3358841, -178.0695038, 187.0637360
4: -72.7263641, 119.6234512, -57.0228500, 95.0533142, -167.7796783, 176.6462860

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B1_A1_B1_A2_B2_A1_B2_B1

### Relational analysis result of NS_B2_B1_A1_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169616, upper bound: 86.5167958
time: 0.73 seconds

## Relational analysis of NS_B2_B1_A1_B1_A2_B2_A1_B2_B2

### Relational analysis result of NS_B2_B1_A1_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169114, upper bound: 86.5171430
time: 0.67 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -35.5479622, 79.2193222, -25.6750126, 57.3344116, -92.8823624, 104.8943176
1: -72.4496384, 118.0211105, -52.6890602, 85.8159485, -158.2655640, 170.7101746
2: -55.8504601, 113.7372742, -40.5704422, 82.6611710, -138.5116272, 154.3077087
3: -85.0804291, 136.2293701, -61.6960068, 99.1195831, -184.1999359, 197.9253845
4: -78.4579697, 130.3020477, -56.9187546, 94.6435776, -173.1015472, 187.2207642

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A1_B1_A2_B2_A2_B1_B1

### Relational analysis result of NS_B2_B1_A1_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5176775, upper bound: 86.5179209
time: 0.72 seconds

## Relational analysis of NS_B2_B1_A1_B1_A2_B2_A2_B1_B2

### Relational analysis result of NS_B2_B1_A1_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5175471, upper bound: 86.5178243
time: 0.72 seconds

## BFS NS instance: NS_B2_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -35.3676109, 78.8030777, -26.0015678, 58.0551300, -93.4227448, 104.8046417
1: -72.0385742, 117.4013977, -53.3338928, 86.9300842, -158.9686432, 170.7352905
2: -55.5525589, 113.1203079, -41.0832253, 83.7237320, -139.2762909, 154.2035370
3: -84.6272049, 135.4969482, -62.4661636, 100.3933868, -185.0205536, 197.9631042
4: -78.0595093, 129.5973816, -57.6199684, 95.8558578, -173.9153595, 187.2173309

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_B1_A1_B1_A2_B2_A2_B2_B1

### Relational analysis result of NS_B2_B1_A1_B1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5139365, upper bound: 86.5143495
time: 0.78 seconds

## Relational analysis of NS_B2_B1_A1_B1_A2_B2_A2_B2_B2

### Relational analysis result of NS_B2_B1_A1_B1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5141556, upper bound: 86.5147645
time: 0.78 seconds

## BFS NS instance: NS_B2_B1_A1_B2_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -24.4952698, 54.2567024, -33.4515381, 74.1282578, -98.6235275, 87.7082214
1: -50.1875725, 81.0191422, -67.9709015, 110.2448883, -160.4324493, 148.9900208
2: -38.6676903, 78.2001114, -52.4670525, 106.3156815, -144.9833679, 130.6671448
3: -58.7932396, 93.5394440, -79.9404144, 127.1589127, -185.9521027, 173.4798431
4: -54.2706947, 89.4193649, -73.8022842, 121.6787109, -175.9494019, 163.2216492

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A1_B1_B1

### Relational analysis result of NS_B2_B1_A1_B2_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5189080, upper bound: 86.5179793
time: 1.04 seconds

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A1_B1_B2

### Relational analysis result of NS_B2_B1_A1_B2_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5189080, upper bound: 86.5179793
time: 0.66 seconds

## BFS NS instance: NS_B2_B1_A1_B2_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -24.4952698, 54.2567024, -35.3980141, 78.8981094, -103.3933716, 89.6547165
1: -50.1875725, 81.0191422, -72.1354904, 117.5395126, -167.7270660, 153.1546021
2: -38.6676903, 78.2001114, -55.6108932, 113.2751160, -151.9428101, 133.8110046
3: -58.7932396, 93.5394440, -84.7186813, 135.6786804, -194.4718628, 178.2581024
4: -54.2706947, 89.4193649, -78.1285858, 129.7758789, -184.0465546, 167.5479126

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A1_B2_B1

### Relational analysis result of NS_B2_B1_A1_B2_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5189080, upper bound: 86.5184509
time: 0.65 seconds

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A1_B2_B2

### Relational analysis result of NS_B2_B1_A1_B2_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5189080, upper bound: 86.5184509
time: 0.95 seconds

## BFS NS instance: NS_B2_B1_A1_B2_A1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -25.2255039, 56.2177696, -33.5268974, 74.1078873, -99.3333893, 89.7446671
1: -51.9640923, 84.0697784, -68.0616684, 110.1852646, -162.1493530, 152.1314392
2: -39.8810005, 80.9348679, -52.5722961, 106.3134995, -146.1944427, 133.5071716
3: -60.6850281, 96.9335785, -80.0830154, 127.1235580, -187.8085938, 177.0165710
4: -55.7992172, 92.6811295, -73.9781723, 121.6077576, -177.4069519, 166.6593018

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A2_A1_B1

### Relational analysis result of NS_B2_B1_A1_B2_A1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5164144, upper bound: 86.5160084
time: 0.72 seconds

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A2_A1_B2

### Relational analysis result of NS_B2_B1_A1_B2_A1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5164144, upper bound: 86.5160084
time: 0.72 seconds

## BFS NS instance: NS_B2_B1_A1_B2_A1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -23.9402428, 52.9656219, -33.9895935, 75.1088104, -99.0490341, 86.9552155
1: -49.0896530, 78.9019241, -69.0205460, 111.6948624, -160.7845001, 147.9224701
2: -37.7739487, 76.1650162, -53.2994270, 107.7945251, -145.5684814, 129.4644470
3: -57.4661217, 91.0278015, -81.2020111, 128.8800964, -186.3462219, 172.2297516
4: -52.9582825, 87.1572952, -74.9941254, 123.2731247, -176.2314148, 162.1513977

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A2_A2_B1

### Relational analysis result of NS_B2_B1_A1_B2_A1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169680, upper bound: 86.5160730
time: 0.81 seconds

## Relational analysis of NS_B2_B1_A1_B2_A1_A1_A2_A2_B2

### Relational analysis result of NS_B2_B1_A1_B2_A1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -86.5169680, upper bound: 86.5160730
time: 0.74 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.26 + 417.76 = 421.02 seconds
