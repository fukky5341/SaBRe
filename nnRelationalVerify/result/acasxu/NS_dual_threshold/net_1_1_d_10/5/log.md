## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 743.673742927666


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-138.5269012, 721.5264282, -138.5269012, 721.5264282, -860.0532837, 860.0533447)
1: (-226.4326935, 857.1390381, -226.4326935, 857.1390381, -1083.5717773, 1083.5716553)
2: (-160.1910706, 887.5496826, -160.1910706, 887.5496826, -1047.7407227, 1047.7406006)
3: (-390.1859741, 752.6910400, -390.1859741, 752.6910400, -1142.8769531, 1142.8769531)
4: (-263.8327942, 761.5472412, -263.8327942, 761.5472412, -1025.3800049, 1025.3800049)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.73 + 1.85 = 2.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -743.6886167, upper bound: 743.6886167

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6819280, upper bound: 743.6817214
time: 0.60 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802824, upper bound: 743.6802824
time: 0.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.38 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 0, lower bound: -743.6819280, upper bound: 743.6817214
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 0, lower bound: -743.6802824, upper bound: 743.6802824

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -134.0998993, 698.8281250, -128.5790863, 670.3025513, -804.4024048, 827.4071655
1: -219.3339996, 830.1954346, -210.4309387, 796.4016113, -1015.7355957, 1040.6263428
2: -155.0493164, 859.5734253, -148.6245575, 824.4069214, -979.4562378, 1008.1979980
3: -377.8995972, 728.8269043, -362.4870300, 699.0101929, -1076.9097900, 1091.3137207
4: -255.3451080, 737.4572144, -244.7495728, 707.2666626, -962.6117554, 982.2067871

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802824, upper bound: 743.6802824
time: 0.66 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802824, upper bound: 743.6802824
time: 0.57 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -132.0244751, 690.9362793, -250.1162415, 1337.2148438, -1469.2392578, 941.0524902
1: -215.8887939, 820.3369141, -412.2479858, 1588.0107422, -1803.8995361, 1232.5849609
2: -152.8052368, 849.8156738, -290.3746338, 1641.2213135, -1794.0263672, 1140.1901855
3: -372.3732300, 719.3096313, -711.9147949, 1393.7313232, -1766.1044922, 1431.2243652
4: -251.5644379, 728.5810547, -478.8499146, 1407.6657715, -1659.2302246, 1207.4309082

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6469194, upper bound: 743.6454741
time: 0.62 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6256300, upper bound: 743.6256300
time: 0.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.04 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 2.04
Output dim: 0, lower bound: -743.6802824, upper bound: 743.6802824
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 2.04
Output dim: 0, lower bound: -743.6802824, upper bound: 743.6802824
NS_B2_A1, status: Status.VERIFIED, split count: 2, time: 2.04
Output dim: 0, lower bound: -743.6469194, upper bound: 743.6454741
NS_B2_A2, status: Status.VERIFIED, split count: 2, time: 2.04
Output dim: 0, lower bound: -743.6256300, upper bound: 743.6256300

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -128.5790863, 670.3025513, -128.5790863, 670.3025513, -798.8816528, 798.8816528
1: -210.4309387, 796.4016113, -210.4309387, 796.4016113, -1006.8324585, 1006.8324585
2: -148.6245575, 824.4069214, -148.6245575, 824.4069214, -973.0314331, 973.0314331
3: -362.4870300, 699.0101929, -362.4870300, 699.0101929, -1061.4970703, 1061.4970703
4: -244.7495728, 707.2666626, -244.7495728, 707.2666626, -952.0162354, 952.0162354

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6751366, upper bound: 743.6741887
time: 0.57 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6730057, upper bound: 743.6736916
time: 0.84 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -249.8228302, 1335.7607422, -128.5790863, 670.3025513, -920.1253662, 1464.3398438
1: -411.7687988, 1586.2872314, -210.4309387, 796.4016113, -1208.1704102, 1796.7181396
2: -290.0353699, 1639.4252930, -148.6245575, 824.4069214, -1114.4422607, 1788.0498047
3: -711.0888672, 1392.1978760, -362.4870300, 699.0101929, -1410.0991211, 1754.6848145
4: -478.2883606, 1406.1068115, -244.7495728, 707.2666626, -1185.5550537, 1650.8562012

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6204047, upper bound: 743.6436335
time: 0.66 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6020628, upper bound: 743.6090481
time: 0.57 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.06 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 0, lower bound: -743.6751366, upper bound: 743.6741887
NS_B1_A1_A2, status: Status.VERIFIED, split count: 3, time: 2.06
Output dim: 0, lower bound: -743.6730057, upper bound: 743.6736916
NS_B1_A2_B1, status: Status.VERIFIED, split count: 3, time: 2.06
Output dim: 0, lower bound: -743.6204047, upper bound: 743.6436335
NS_B1_A2_B2, status: Status.VERIFIED, split count: 3, time: 2.06
Output dim: 0, lower bound: -743.6020628, upper bound: 743.6090481

## BFS NS instance: NS_B1_A1_A1

### Backsubstitution after applying NS history:
0: -123.4315948, 642.8097534, -128.5790863, 670.3025513, -793.7341309, 771.3887939
1: -201.7657318, 763.8026123, -210.4309387, 796.4016113, -998.1673584, 974.2334595
2: -142.5318909, 790.5067139, -148.6245575, 824.4069214, -966.9387817, 939.1312256
3: -347.4673157, 670.4584961, -362.4870300, 699.0101929, -1046.4772949, 1032.9454346
4: -234.7127075, 678.4046021, -244.7495728, 707.2666626, -941.9793701, 923.1541748

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6844180, upper bound: 743.6835588
time: 0.67 seconds

## Relational analysis of NS_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6844465, upper bound: 743.6841092
time: 0.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.02 seconds
NS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -743.6844180, upper bound: 743.6835588
NS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -743.6844465, upper bound: 743.6841092

## BFS NS instance: NS_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -120.5338745, 627.5079346, -121.0235214, 630.4956055, -751.0294800, 748.5314331
1: -196.9615326, 745.5234375, -197.9168854, 748.8142090, -945.7756348, 943.4403076
2: -139.1340790, 771.7973633, -139.7664032, 775.9002075, -915.0342407, 911.5637817
3: -339.1861877, 654.1229858, -340.9161682, 656.4567261, -995.6429443, 995.0391846
4: -229.0725861, 662.1480713, -230.0600586, 664.9388428, -894.0114136, 892.2080688

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_A1_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6716896, upper bound: 743.6719372
time: 0.54 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6840162, upper bound: 743.6830347
time: 0.79 seconds

## BFS NS instance: NS_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -121.2424164, 631.3088989, -126.0463181, 658.2313843, -779.4738159, 757.3552246
1: -198.1676636, 750.1177979, -206.1511536, 781.9909058, -980.1585693, 956.2687988
2: -139.9758759, 776.3911133, -145.6629333, 809.3348389, -949.3106079, 922.0540771
3: -341.2210693, 658.3440552, -355.0702209, 685.9469604, -1027.1679688, 1013.4143066
4: -230.4693451, 666.2149658, -239.8355560, 693.8856812, -924.3549194, 906.0504761

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_A1_A1_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6808308, upper bound: 743.6812716
time: 0.62 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6811787, upper bound: 743.6806375
time: 0.78 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.26 seconds
NS_B1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 0, lower bound: -743.6716896, upper bound: 743.6719372
NS_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -743.6840162, upper bound: 743.6830347
NS_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -743.6808308, upper bound: 743.6812716
NS_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -743.6811787, upper bound: 743.6806375

## BFS NS instance: NS_B1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -114.4795074, 596.7202148, -117.9355927, 614.6185913, -729.0980835, 714.6558228
1: -186.9379578, 708.6897583, -192.8064270, 729.8828125, -916.8208008, 901.4960938
2: -132.1466522, 734.1527100, -136.1939697, 756.4288330, -888.5754395, 870.3466797
3: -321.9257812, 621.2741089, -332.1548767, 639.6998291, -961.6254272, 953.4288330
4: -217.5351105, 629.0924072, -224.1878204, 648.0467529, -865.5818481, 853.2802124

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A1_B1_A2_B1

### Relational analysis result of NS_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6767918, upper bound: 743.6769715
time: 0.64 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_B2

### Relational analysis result of NS_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6839879, upper bound: 743.6829580
time: 0.59 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -115.7838440, 601.9401855, -124.1502380, 648.0225220, -763.8063354, 726.0904541
1: -189.1952209, 715.1301880, -203.0335388, 769.8319702, -959.0271606, 918.1636963
2: -133.6273346, 740.5656128, -143.4587402, 796.8896484, -930.5169678, 884.0243530
3: -325.7557373, 627.3670044, -349.6967773, 675.1855469, -1000.9412842, 977.0636597
4: -219.9403534, 635.1920166, -236.1842346, 683.0995483, -903.0399170, 871.3762207

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A1_B2_A1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6629408, upper bound: 743.6628064
time: 0.80 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B1_A1_A1_B2_A1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6808308, upper bound: 743.6802545
time: 0.60 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6808308, upper bound: 743.6806375
time: 0.52 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -120.6389847, 628.4600220, -123.7459946, 645.8974609, -766.5364380, 752.2058716
1: -197.0671539, 746.4927368, -202.3405914, 767.3287354, -964.3958740, 948.8333130
2: -139.2612457, 772.9766846, -142.9669952, 794.1891479, -933.4503784, 915.9436646
3: -339.5575867, 654.7389526, -348.4742432, 672.9882202, -1012.5457764, 1003.2131958
4: -229.2952728, 662.8297119, -235.3658447, 680.8421631, -910.1374512, 898.1954956

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A1_B2_A2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6740953, upper bound: 743.6727747
time: 0.88 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B1_A1_A1_B2_A2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6811787, upper bound: 743.6802545
time: 0.79 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6811787, upper bound: 743.6806375
time: 0.70 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.88 seconds
NS_B1_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -743.6767918, upper bound: 743.6769715
NS_B1_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -743.6839879, upper bound: 743.6829580
NS_B1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -743.6808308, upper bound: 743.6802545
NS_B1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -743.6808308, upper bound: 743.6806375
NS_B1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -743.6811787, upper bound: 743.6802545
NS_B1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -743.6811787, upper bound: 743.6806375

## BFS NS instance: NS_B1_A1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -103.4658051, 539.9147339, -105.6933060, 551.2001953, -654.6659546, 645.6080322
1: -168.8389130, 640.4889526, -172.6381073, 653.8652344, -822.7040405, 813.1269531
2: -119.4442673, 664.7098389, -122.0820847, 678.6158447, -798.0601196, 786.7918701
3: -290.9145203, 560.3219604, -297.6408081, 571.7973633, -862.7119141, 857.9626465
4: -196.5770874, 568.5398560, -200.8965607, 580.6843872, -777.2614746, 769.4364014

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_A1_A1_B1_A2_B1_B1

### Relational analysis result of NS_B1_A1_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6610836, upper bound: 743.6612969
time: 0.79 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_B1_B2

### Relational analysis result of NS_B1_A1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6610836, upper bound: 743.6752438
time: 0.67 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -114.1868591, 595.1969604, -117.4393463, 612.0337524, -726.2205811, 712.6362915
1: -186.4616547, 706.8694458, -192.0006256, 726.7934570, -913.2551270, 898.8700562
2: -131.8097992, 732.2888794, -135.6225739, 753.2695312, -885.0792847, 867.9114380
3: -321.0958862, 619.6533203, -330.7568665, 636.9489136, -958.0446777, 950.4101562
4: -216.9765472, 627.4667969, -223.2399139, 645.2973633, -862.2739258, 850.7066650

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B1

### Relational analysis result of NS_B1_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6838964, upper bound: 743.6829580
time: 0.55 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B2

### Relational analysis result of NS_B1_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6838964, upper bound: 743.6829580
time: 0.55 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -115.7838440, 601.9401855, -120.5769043, 628.8318481, -744.6156616, 722.5170898
1: -189.1952209, 715.1301880, -197.1554260, 746.9666138, -936.1617432, 912.2855225
2: -133.6273346, 740.5656128, -139.3046417, 773.5567017, -907.1840210, 879.8702393
3: -325.7557373, 627.3670044, -339.5640564, 654.9341431, -980.6898804, 966.9309692
4: -219.9403534, 635.1920166, -229.3020020, 662.8068237, -882.7471924, 864.4940186

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6474754, upper bound: 743.6496380
time: 0.72 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6808145, upper bound: 743.6811954
time: 0.60 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -115.7838440, 601.9401855, -125.4167252, 655.2555542, -771.0393677, 727.3569336
1: -189.1952209, 715.1301880, -205.0163879, 778.2237549, -967.4188843, 920.1465454
2: -133.6273346, 740.5656128, -144.9149780, 805.7722168, -939.3995361, 885.4805908
3: -325.7557373, 627.3670044, -353.3381653, 682.1816406, -1007.9373779, 980.7050781
4: -219.9403534, 635.1920166, -238.6045380, 690.3400879, -910.2804565, 873.7965698

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6474754, upper bound: 743.6496380
time: 0.58 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6808145, upper bound: 743.6811954
time: 0.94 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -120.6389847, 628.4600220, -120.5769043, 628.8318481, -749.4708252, 749.0369263
1: -197.0671539, 746.4927368, -197.1554260, 746.9666138, -944.0337524, 943.6480713
2: -139.2612457, 772.9766846, -139.3046417, 773.5567017, -912.8179321, 912.2813110
3: -339.5575867, 654.7389526, -339.5640564, 654.9341431, -994.4916992, 994.3029175
4: -229.2952728, 662.8297119, -229.3020020, 662.8068237, -892.1021118, 892.1317139

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6811787, upper bound: 743.6802545
time: 0.84 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6811787, upper bound: 743.6802545
time: 0.61 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -120.6389847, 628.4600220, -125.4212265, 655.2789917, -775.9179688, 753.8811646
1: -197.0671539, 746.4927368, -205.0236664, 778.2514648, -975.3185425, 951.5162354
2: -139.2612457, 772.9766846, -144.9201355, 805.8010254, -945.0622559, 917.8967896
3: -339.5575867, 654.7389526, -353.3507690, 682.2062988, -1021.7639160, 1008.0897217
4: -229.2952728, 662.8297119, -238.6129303, 690.3650513, -919.6603394, 901.4426270

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6811787, upper bound: 743.6802545
time: 0.65 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6811787, upper bound: 743.6802545
time: 0.70 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.38 seconds
NS_B1_A1_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 2.38
Output dim: 0, lower bound: -743.6610836, upper bound: 743.6612969
NS_B1_A1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.38
Output dim: 0, lower bound: -743.6610836, upper bound: 743.6752438
NS_B1_A1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.38
Output dim: 0, lower bound: -743.6838964, upper bound: 743.6829580
NS_B1_A1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.38
Output dim: 0, lower bound: -743.6838964, upper bound: 743.6829580
NS_B1_A1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.38
Output dim: 0, lower bound: -743.6474754, upper bound: 743.6496380
NS_B1_A1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.38
Output dim: 0, lower bound: -743.6808145, upper bound: 743.6811954
NS_B1_A1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.38
Output dim: 0, lower bound: -743.6474754, upper bound: 743.6496380
NS_B1_A1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.38
Output dim: 0, lower bound: -743.6808145, upper bound: 743.6811954
NS_B1_A1_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.38
Output dim: 0, lower bound: -743.6811787, upper bound: 743.6802545
NS_B1_A1_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.38
Output dim: 0, lower bound: -743.6811787, upper bound: 743.6802545
NS_B1_A1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.38
Output dim: 0, lower bound: -743.6811787, upper bound: 743.6802545
NS_B1_A1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.38
Output dim: 0, lower bound: -743.6811787, upper bound: 743.6802545

## BFS NS instance: NS_B1_A1_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -103.4658051, 539.9147339, -102.1606293, 532.7718506, -636.2376709, 642.0751953
1: -168.8389130, 640.4889526, -166.7559814, 631.8485107, -800.6873169, 807.2449341
2: -119.4442673, 664.7098389, -118.0134430, 656.0633545, -775.5076294, 782.7231445
3: -290.9145203, 560.3219604, -287.5096741, 552.3092041, -843.2237549, 847.8314209
4: -196.5770874, 568.5398560, -194.1813049, 561.0806274, -757.6577148, 762.7211914

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_A1_B1_A2_B1_B2_B1

### Relational analysis result of NS_B1_A1_A1_B1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6610836, upper bound: 743.6716887
time: 0.84 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_B1_B2_B2

### Relational analysis result of NS_B1_A1_A1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6610836, upper bound: 743.6752438
time: 0.70 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -114.1868591, 595.1969604, -112.4364548, 585.2879639, -699.4748535, 707.6334229
1: -186.4616547, 706.8694458, -183.5759125, 695.1024170, -881.5640869, 890.4453735
2: -131.8097992, 732.2888794, -129.6961365, 720.2097168, -852.0194702, 861.9849854
3: -321.0958862, 619.6533203, -316.1567993, 609.2200317, -930.3157959, 935.8101196
4: -216.9765472, 627.4667969, -213.4718170, 617.2585449, -834.2351074, 840.9385986

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B1_B1

### Relational analysis result of NS_B1_A1_A1_B1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6466981, upper bound: 743.6519511
time: 0.62 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B1_B2

### Relational analysis result of NS_B1_A1_A1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6466981, upper bound: 743.6829022
time: 0.64 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -114.1868591, 595.1969604, -116.0502167, 605.4920044, -719.6788330, 711.2471924
1: -186.4616547, 706.8694458, -189.6126404, 718.5642700, -905.0259399, 896.4820557
2: -131.8097992, 732.2888794, -133.9753113, 745.2537842, -877.0635376, 866.2641602
3: -321.0958862, 619.6533203, -326.6847534, 629.0847778, -950.1805420, 946.3380737
4: -216.9765472, 627.4667969, -220.4600983, 637.7439575, -854.7205200, 847.9268799

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B2_B1

### Relational analysis result of NS_B1_A1_A1_B1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6466981, upper bound: 743.6519511
time: 0.65 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B2_B2

### Relational analysis result of NS_B1_A1_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6466981, upper bound: 743.6829022
time: 0.54 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -115.2467957, 599.1481934, -120.1617203, 626.6961060, -741.9428711, 719.3099365
1: -188.3226318, 711.7914429, -196.4811554, 744.4093018, -932.7318726, 908.2725220
2: -133.0089264, 737.1618652, -138.8274841, 770.9506836, -903.9595947, 875.9892578
3: -324.2524109, 624.3947754, -338.4010620, 652.6517944, -976.9041748, 962.7957764
4: -218.9150238, 632.2325439, -228.5098114, 660.5354614, -879.4504395, 860.7421875

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6830375, upper bound: 743.6828330
time: 0.57 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6830375, upper bound: 743.6828330
time: 0.71 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -115.2467957, 599.1481934, -124.9079132, 652.6987305, -767.9455566, 724.0560913
1: -188.3226318, 711.7914429, -204.1948853, 775.1463013, -963.4689331, 915.9862671
2: -133.0089264, 737.1618652, -144.3336334, 802.6262817, -935.6351929, 881.4954834
3: -324.2524109, 624.3947754, -351.9063416, 679.4227905, -1003.6751099, 976.3011475
4: -218.9150238, 632.2325439, -237.6408844, 687.5870972, -906.5020752, 869.8734131

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6808145, upper bound: 743.6811954
time: 0.69 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6808145, upper bound: 743.6811954
time: 0.61 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -120.6389847, 628.4600220, -115.5170975, 601.9134521, -722.5524292, 743.9770508
1: -197.0671539, 746.4927368, -188.6524506, 715.0547485, -912.1218872, 935.1451416
2: -139.2612457, 772.9766846, -133.3189087, 740.3151245, -879.5763550, 906.2955933
3: -339.5575867, 654.7389526, -324.8376160, 626.9719238, -966.5295410, 979.5764160
4: -229.2952728, 662.8297119, -219.4403839, 634.5384521, -863.8336792, 882.2700806

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6808450, upper bound: 743.6795255
time: 0.73 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6807873, upper bound: 743.6796930
time: 0.62 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -120.6389847, 628.4600220, -118.9941788, 621.0119629, -741.6509399, 747.4541626
1: -197.0671539, 746.4927368, -194.4658661, 737.2043457, -934.2714844, 940.9585571
2: -139.2612457, 772.9766846, -137.4392700, 764.0627441, -903.3239746, 910.4158936
3: -339.5575867, 654.7389526, -334.9507446, 645.9365845, -985.4941406, 989.6896973
4: -229.2952728, 662.8297119, -226.1694336, 654.0452271, -883.3405151, 888.9991455

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6808450, upper bound: 743.6795255
time: 0.74 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6807873, upper bound: 743.6796930
time: 0.64 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -120.6389847, 628.4600220, -120.2601471, 627.7479858, -748.3869629, 748.7201538
1: -197.0671539, 746.4927368, -196.3567047, 745.6128540, -942.6799316, 942.8493652
2: -139.2612457, 772.9766846, -138.8194427, 771.8737183, -911.1349487, 911.7961426
3: -339.5575867, 654.7389526, -338.3160400, 653.6276855, -993.1853027, 993.0549927
4: -229.2952728, 662.8297119, -228.5553894, 661.4656982, -890.7609863, 891.3850098

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6801178, upper bound: 743.6797876
time: 0.91 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802120, upper bound: 743.6795698
time: 0.59 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -120.6389847, 628.4600220, -124.2415695, 648.8683472, -769.5073242, 752.7015991
1: -197.0671539, 746.4927368, -202.9208984, 770.4276733, -967.4948120, 949.4136353
2: -139.2612457, 772.9766846, -143.4839935, 798.3078003, -937.5690308, 916.4606934
3: -339.5575867, 654.7389526, -349.7830200, 675.0649414, -1014.6225586, 1004.5219116
4: -229.2952728, 662.8297119, -236.1858521, 683.4622803, -912.7575684, 899.0155640

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6801178, upper bound: 743.6797876
time: 0.79 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802120, upper bound: 743.6795698
time: 0.64 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.88 seconds
NS_B1_A1_A1_B1_A2_B1_B2_B1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -743.6610836, upper bound: 743.6716887
NS_B1_A1_A1_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -743.6610836, upper bound: 743.6752438
NS_B1_A1_A1_B1_A2_B2_B1_B1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -743.6466981, upper bound: 743.6519511
NS_B1_A1_A1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -743.6466981, upper bound: 743.6829022
NS_B1_A1_A1_B1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -743.6466981, upper bound: 743.6519511
NS_B1_A1_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -743.6466981, upper bound: 743.6829022
NS_B1_A1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -743.6830375, upper bound: 743.6828330
NS_B1_A1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -743.6830375, upper bound: 743.6828330
NS_B1_A1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -743.6808145, upper bound: 743.6811954
NS_B1_A1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -743.6808145, upper bound: 743.6811954
NS_B1_A1_A1_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -743.6808450, upper bound: 743.6795255
NS_B1_A1_A1_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -743.6807873, upper bound: 743.6796930
NS_B1_A1_A1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -743.6808450, upper bound: 743.6795255
NS_B1_A1_A1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -743.6807873, upper bound: 743.6796930
NS_B1_A1_A1_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -743.6801178, upper bound: 743.6797876
NS_B1_A1_A1_B2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -743.6802120, upper bound: 743.6795698
NS_B1_A1_A1_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -743.6801178, upper bound: 743.6797876
NS_B1_A1_A1_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -743.6802120, upper bound: 743.6795698

## BFS NS instance: NS_B1_A1_A1_B1_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -103.4658051, 539.9147339, -101.0190201, 527.0537109, -630.5195312, 640.9336548
1: -168.8389130, 640.4889526, -164.6903229, 624.9317017, -793.7705688, 805.1791992
2: -119.4442673, 664.7098389, -116.6570587, 648.9560547, -768.4003296, 781.3668213
3: -290.9145203, 560.3219604, -284.0819397, 545.7727051, -836.6872559, 844.4037476
4: -196.5770874, 568.5398560, -191.9504242, 554.8154907, -751.3925781, 760.4902954

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A1_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A1_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B1_A2_B1_B2_B2_A1

### Relational analysis result of NS_B1_A1_A1_B1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6728254, upper bound: 743.6752438
time: 0.64 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_B1_B2_B2_A2

### Relational analysis result of NS_B1_A1_A1_B1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6728254, upper bound: 743.6752438
time: 0.62 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -114.1868591, 595.1969604, -109.4853592, 570.5167847, -684.7036133, 704.6821899
1: -186.4616547, 706.8694458, -178.6757812, 677.3356934, -863.7973633, 885.5452271
2: -131.8097992, 732.2888794, -126.3032150, 702.2443237, -834.0540771, 858.5921021
3: -321.0958862, 619.6533203, -307.7005615, 593.2511597, -914.3469238, 927.3538818
4: -216.9765472, 627.4667969, -207.8475342, 601.2461548, -818.2227173, 835.3143311

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B1_B2_B1

### Relational analysis result of NS_B1_A1_A1_B1_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6765068, upper bound: 743.6822904
time: 0.64 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B1_B2_B2

### Relational analysis result of NS_B1_A1_A1_B1_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6744923, upper bound: 743.6813247
time: 0.64 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -114.1868591, 595.1969604, -112.7105637, 588.9992065, -703.1860352, 707.9075317
1: -186.4616547, 706.8694458, -184.0625458, 698.7335815, -885.1952515, 890.9320068
2: -131.8097992, 732.2888794, -130.1325226, 724.9702759, -856.7800293, 862.4213867
3: -321.0958862, 619.6533203, -317.1809998, 611.1693115, -932.2650757, 936.8343506
4: -216.9765472, 627.4667969, -214.1115265, 619.7582397, -836.7348022, 841.5783081

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B2_B2_A1

### Relational analysis result of NS_B1_A1_A1_B1_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6437472, upper bound: 743.6696741
time: 0.64 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B2_B2_A2

### Relational analysis result of NS_B1_A1_A1_B1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6437472, upper bound: 743.6829015
time: 1.02 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -115.2467957, 599.1481934, -115.1039505, 599.7886353, -715.0354004, 714.2521362
1: -188.3226318, 711.7914429, -187.9810181, 712.5099487, -900.8325806, 899.7724609
2: -133.0089264, 737.1618652, -132.8440552, 737.7230835, -870.7319946, 870.0059204
3: -324.2524109, 624.3947754, -323.6802063, 624.7008057, -948.9531250, 948.0748901
4: -218.9150238, 632.2325439, -218.6524048, 632.2791748, -851.1942139, 850.8849487

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2_B1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6817297, upper bound: 743.6818365
time: 0.71 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2_B1_B2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6814881, upper bound: 743.6813413
time: 0.60 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -115.2467957, 599.1481934, -118.7095108, 619.5377197, -734.7845459, 717.8577271
1: -188.3226318, 711.7914429, -194.0026398, 735.4414673, -923.7640381, 905.7940674
2: -133.0089264, 737.1618652, -137.1115875, 762.2473145, -895.2562256, 874.2733765
3: -324.2524109, 624.3947754, -334.1413879, 644.3536987, -968.6060181, 958.5360718
4: -218.9150238, 632.2325439, -225.6249390, 652.4591675, -871.3741455, 857.8574219

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6821810, upper bound: 743.6828330
time: 0.83 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6821810, upper bound: 743.6822388
time: 0.66 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -115.2467957, 599.1481934, -119.7854767, 625.3579712, -740.6047363, 718.9336548
1: -188.3226318, 711.7914429, -195.5922241, 742.7462158, -931.0688477, 907.3836670
2: -133.0089264, 737.1618652, -138.2784119, 768.9310913, -901.9400024, 875.4403076
3: -324.2524109, 624.3947754, -336.9815063, 651.0618286, -975.3142090, 961.3762817
4: -218.9150238, 632.2325439, -227.6591034, 658.8934937, -877.8084106, 859.8916016

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6794594, upper bound: 743.6803775
time: 0.59 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B1_B2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6797694, upper bound: 743.6805269
time: 0.63 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -115.2467957, 599.1481934, -123.5963364, 645.7241821, -760.9709473, 722.7445068
1: -188.3226318, 711.7914429, -201.8932343, 766.6088257, -954.9314575, 913.6846924
2: -133.0089264, 737.1618652, -142.7499542, 794.4553223, -927.4642334, 879.9118042
3: -324.2524109, 624.3947754, -347.9926453, 671.6033325, -995.8556519, 972.3873901
4: -218.9150238, 632.2325439, -234.9583740, 680.0612793, -898.9761963, 867.1907959

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6794594, upper bound: 743.6803775
time: 0.56 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B2_B2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6797694, upper bound: 743.6805269
time: 0.64 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -110.2316360, 573.4473267, -113.4503937, 591.0851440, -701.3167725, 686.8977051
1: -179.9872742, 680.9829102, -185.3094635, 702.1433105, -882.1306152, 866.2923584
2: -127.1333618, 705.6771240, -130.9314880, 727.1167603, -854.2500610, 836.6085815
3: -310.2410278, 597.0095825, -319.0646973, 615.5219727, -925.7629395, 916.0742798
4: -209.2506409, 604.7850952, -215.4860687, 623.0761719, -832.3267822, 820.2711792

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6799250, upper bound: 743.6792656
time: 0.59 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1_A1_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6800803, upper bound: 743.6790155
time: 0.69 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -115.5850449, 602.0866699, -112.8870316, 588.1751099, -703.7600098, 714.9736938
1: -188.8125610, 715.0158081, -184.3446350, 698.6599121, -887.4724731, 899.3603516
2: -133.4127655, 740.6810913, -130.2725067, 723.5343628, -856.9470825, 870.9536133
3: -325.2523499, 626.8279419, -317.3878479, 612.4406738, -937.6929321, 944.2156982
4: -219.5975342, 634.8402100, -214.3972168, 619.9763794, -839.5739136, 849.2373657

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1_A2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6809765, upper bound: 743.6799367
time: 0.62 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1_A2_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6809765, upper bound: 743.6799367
time: 0.55 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -110.2316360, 573.4473267, -116.9270172, 610.1947021, -720.4263306, 690.3743286
1: -179.9872742, 680.9829102, -191.1232300, 724.3070068, -904.2943115, 872.1061401
2: -127.1333618, 705.6771240, -135.0494843, 750.8228760, -877.9561157, 840.7266235
3: -310.2410278, 597.0095825, -329.1664429, 634.4669189, -944.7078857, 926.1760254
4: -209.2506409, 604.7850952, -222.2122650, 642.5473022, -851.7979126, 826.9973755

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802132, upper bound: 743.6791657
time: 0.67 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782814, upper bound: 743.6764188
time: 0.62 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -115.5850449, 602.0866699, -116.3378983, 607.2179565, -722.8029785, 718.4244995
1: -188.8125610, 715.0158081, -190.1071167, 720.7265625, -909.5391235, 905.1228638
2: -133.4127655, 740.6810913, -134.3647461, 747.1541138, -880.5668335, 875.0458374
3: -325.2523499, 626.8279419, -327.4189148, 631.2935181, -956.5457764, 954.2468262
4: -219.5975342, 634.8402100, -221.0815125, 639.3684082, -858.9659424, 855.9216919

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6807873, upper bound: 743.6796930
time: 0.60 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A2_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6807873, upper bound: 743.6796930
time: 0.62 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -118.4224930, 616.8804932, -109.1163712, 568.7373657, -687.1597900, 725.9968262
1: -193.4773560, 732.6821899, -178.0852661, 675.3342896, -868.8115234, 910.7674561
2: -136.7034149, 758.7997437, -125.8366547, 699.7525635, -836.4559937, 884.6364136
3: -333.3541260, 642.4932251, -306.9655457, 591.7061157, -925.0602417, 949.4586792
4: -225.0549927, 650.5684204, -207.0895386, 599.2673950, -824.3223267, 857.6579590

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6586062, upper bound: 743.6655220
time: 0.61 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6586062, upper bound: 743.6804523
time: 0.68 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -117.9529953, 614.4614868, -115.2489243, 601.5029297, -719.4559326, 729.7103271
1: -192.6770630, 729.7847290, -188.1741638, 714.2922974, -906.9693604, 917.9588623
2: -136.1514893, 755.8343506, -133.0243378, 739.7500610, -875.9015503, 888.8586426
3: -331.9507751, 639.9146118, -324.1338501, 625.9076538, -957.8583374, 964.0484619
4: -224.1399536, 647.9645996, -218.9506683, 633.6780396, -857.8179932, 866.9151611

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6804587, upper bound: 743.6798105
time: 0.59 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6804587, upper bound: 743.6799367
time: 0.69 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -118.4224930, 616.8804932, -112.8706055, 588.8263550, -707.2488403, 729.7510986
1: -193.4773560, 732.6821899, -184.2608795, 698.7004395, -892.1776733, 916.9429932
2: -136.7034149, 758.7997437, -130.2121124, 724.6369019, -861.3402710, 889.0118408
3: -333.3541260, 642.4932251, -317.7404175, 611.6794434, -945.0335693, 960.2335815
4: -225.0549927, 650.5684204, -214.2491302, 619.7327271, -844.7877197, 864.8173828

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6801178, upper bound: 743.6795198
time: 0.57 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6801178, upper bound: 743.6795698
time: 0.61 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -117.9529953, 614.4614868, -119.2862473, 623.0567017, -741.0096436, 733.7477417
1: -192.6770630, 729.7847290, -194.8324585, 739.5347290, -932.2116699, 924.6170654
2: -136.1514893, 755.8343506, -137.7608185, 766.6956177, -902.8471069, 893.5951538
3: -331.9507751, 639.9146118, -335.7662354, 647.6977539, -979.6483154, 975.6808472
4: -224.1399536, 647.9645996, -226.7017059, 656.0482178, -880.1881104, 874.6663208

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802120, upper bound: 743.6795198
time: 0.60 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802120, upper bound: 743.6795698
time: 0.65 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 2.13 seconds
NS_B1_A1_A1_B1_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6728254, upper bound: 743.6752438
NS_B1_A1_A1_B1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6728254, upper bound: 743.6752438
NS_B1_A1_A1_B1_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6765068, upper bound: 743.6822904
NS_B1_A1_A1_B1_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6744923, upper bound: 743.6813247
NS_B1_A1_A1_B1_A2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6437472, upper bound: 743.6696741
NS_B1_A1_A1_B1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6437472, upper bound: 743.6829015
NS_B1_A1_A1_B2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6817297, upper bound: 743.6818365
NS_B1_A1_A1_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6814881, upper bound: 743.6813413
NS_B1_A1_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6821810, upper bound: 743.6828330
NS_B1_A1_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6821810, upper bound: 743.6822388
NS_B1_A1_A1_B2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6794594, upper bound: 743.6803775
NS_B1_A1_A1_B2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6797694, upper bound: 743.6805269
NS_B1_A1_A1_B2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6794594, upper bound: 743.6803775
NS_B1_A1_A1_B2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6797694, upper bound: 743.6805269
NS_B1_A1_A1_B2_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6799250, upper bound: 743.6792656
NS_B1_A1_A1_B2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6800803, upper bound: 743.6790155
NS_B1_A1_A1_B2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6809765, upper bound: 743.6799367
NS_B1_A1_A1_B2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6809765, upper bound: 743.6799367
NS_B1_A1_A1_B2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6802132, upper bound: 743.6791657
NS_B1_A1_A1_B2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6782814, upper bound: 743.6764188
NS_B1_A1_A1_B2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6807873, upper bound: 743.6796930
NS_B1_A1_A1_B2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6807873, upper bound: 743.6796930
NS_B1_A1_A1_B2_A2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6586062, upper bound: 743.6655220
NS_B1_A1_A1_B2_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6586062, upper bound: 743.6804523
NS_B1_A1_A1_B2_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6804587, upper bound: 743.6798105
NS_B1_A1_A1_B2_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6804587, upper bound: 743.6799367
NS_B1_A1_A1_B2_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6801178, upper bound: 743.6795198
NS_B1_A1_A1_B2_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6801178, upper bound: 743.6795698
NS_B1_A1_A1_B2_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6802120, upper bound: 743.6795198
NS_B1_A1_A1_B2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.13
Output dim: 0, lower bound: -743.6802120, upper bound: 743.6795698

## BFS NS instance: NS_B1_A1_A1_B1_A2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -99.4743958, 518.8283081, -101.0190201, 527.0537109, -626.5279541, 619.8472900
1: -162.2768097, 615.2236328, -164.6903229, 624.9317017, -787.2084961, 779.9139404
2: -114.7789688, 638.9635010, -116.6570587, 648.9560547, -763.7350464, 755.6205444
3: -279.5502930, 537.7435303, -284.0819397, 545.7727051, -825.3229980, 821.8253784
4: -188.8437958, 546.1507568, -191.9504242, 554.8154907, -743.6593018, 738.1011963

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A1_B1_A2_B1_B2_B2_A1_A1

### Relational analysis result of NS_B1_A1_A1_B1_A2_B1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6701778, upper bound: 743.6702882
time: 0.67 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_B1_B2_B2_A1_A2

### Relational analysis result of NS_B1_A1_A1_B1_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6701778, upper bound: 743.6752438
time: 0.85 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -102.1382980, 534.4179077, -101.0190201, 527.0537109, -629.1919556, 635.4368896
1: -166.6064606, 634.0441895, -164.6903229, 624.9317017, -791.5381470, 798.7344971
2: -117.9231415, 657.6874390, -116.6570587, 648.9560547, -766.8792114, 774.3444824
3: -286.9412842, 554.4375000, -284.0819397, 545.7727051, -832.7139893, 838.5194092
4: -194.0457458, 562.1912231, -191.9504242, 554.8154907, -748.8612061, 754.1416626

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_B1_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A1_B1_A2_B1_B2_B2_A2_A1

### Relational analysis result of NS_B1_A1_A1_B1_A2_B1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6701778, upper bound: 743.6702882
time: 0.58 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_B1_B2_B2_A2_A2

### Relational analysis result of NS_B1_A1_A1_B1_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6701778, upper bound: 743.6752438
time: 0.66 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A2_B2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -111.9881668, 583.6520996, -98.7056732, 514.0566406, -626.0447998, 682.3577881
1: -182.9001465, 693.1135864, -161.0493622, 609.8571777, -792.7572632, 854.1629028
2: -129.2668152, 718.2088013, -113.7598038, 633.0017090, -762.2685547, 831.9686279
3: -314.9401550, 607.4746094, -277.3782959, 533.6101074, -848.5501099, 884.8529053
4: -212.7627258, 615.2485352, -187.0927734, 541.3200684, -754.0827637, 802.3413086

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B1_B2_B1_A1

### Relational analysis result of NS_B1_A1_A1_B1_A2_B2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6678047, upper bound: 743.6684864
time: 0.84 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B1_B2_B1_A2

### Relational analysis result of NS_B1_A1_A1_B1_A2_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6678047, upper bound: 743.6821405
time: 0.64 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A2_B2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -111.4155960, 580.8088379, -104.5356674, 544.6671143, -656.0827026, 685.3444824
1: -181.9226837, 689.6831665, -170.6077576, 646.4310303, -828.3536987, 860.2908936
2: -128.6031799, 714.7100830, -120.5796356, 670.6389771, -799.2421875, 835.2897339
3: -313.2512207, 604.3911133, -293.6960144, 565.8764038, -879.1276245, 898.0871582
4: -211.6607056, 612.1766357, -198.3574982, 573.8297119, -785.4903564, 810.5340576

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B1_B2_B2_A1

### Relational analysis result of NS_B1_A1_A1_B1_A2_B2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6675944, upper bound: 743.6680998
time: 0.75 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B1_B2_B2_A2

### Relational analysis result of NS_B1_A1_A1_B1_A2_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6675944, upper bound: 743.6813201
time: 0.63 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A2_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -114.0016403, 594.2327881, -112.7105637, 588.9992065, -703.0008545, 706.9433594
1: -186.1607513, 705.7164917, -184.0625458, 698.7335815, -884.8943481, 889.7790527
2: -131.5966034, 731.1126709, -130.1325226, 724.9702759, -856.5668335, 861.2451782
3: -320.5764160, 618.6268311, -317.1809998, 611.1693115, -931.7456665, 935.8078613
4: -216.6230164, 626.4434814, -214.1115265, 619.7582397, -836.3812256, 840.5549927

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B2_B2_A2_A1

### Relational analysis result of NS_B1_A1_A1_B1_A2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6694701, upper bound: 743.6829011
time: 0.82 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_B2_B2_B2_A2_A2

### Relational analysis result of NS_B1_A1_A1_B1_A2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6694701, upper bound: 743.6829015
time: 0.77 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -113.3904114, 589.4231567, -104.1653366, 542.1268921, -655.5173340, 693.5885010
1: -185.3195343, 700.2048340, -170.0540771, 643.6420288, -828.9615479, 870.2589111
2: -130.8647766, 725.2958374, -120.1036377, 667.2286987, -798.0934448, 845.3994751
3: -319.0674438, 614.1246948, -292.8795776, 563.9747925, -883.0422363, 907.0041504
4: -215.3625641, 621.9359741, -197.5836029, 571.2769775, -786.6395264, 819.5195923

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6805599, upper bound: 743.6813627
time: 0.60 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2_B1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6805599, upper bound: 743.6818319
time: 0.63 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -112.4720993, 584.7092896, -109.8714752, 572.4594727, -684.9315186, 694.5806885
1: -183.7790375, 694.5543823, -179.4183350, 679.8875732, -863.6666260, 873.9727173
2: -129.7962646, 719.5155029, -126.7875214, 704.3443604, -834.1406250, 846.3030396
3: -316.3895569, 609.0925903, -308.8663635, 595.7873535, -912.1768799, 917.9589844
4: -213.5906372, 616.8994751, -208.6171112, 603.3142090, -816.9047852, 825.5165405

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6814881, upper bound: 743.6814881
time: 0.60 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6814881, upper bound: 743.6814881
time: 0.63 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -110.2149658, 572.6878052, -118.7095108, 619.5377197, -729.7526855, 691.3973389
1: -179.9665833, 680.0687866, -194.0026398, 735.4414673, -915.4080811, 874.0714111
2: -127.0998230, 705.0490723, -137.1115875, 762.2473145, -889.3471680, 842.1605225
3: -309.8979492, 595.9207153, -334.1413879, 644.3536987, -954.2516479, 930.0620117
4: -209.1191559, 604.0726929, -225.6249390, 652.4591675, -861.5781860, 829.6975708

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2_B2_A1_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6820538, upper bound: 743.6815340
time: 0.76 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2_B2_A1_A2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6808347, upper bound: 743.6807740
time: 0.59 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -114.9268951, 598.8602905, -118.7095108, 619.5377197, -734.4645996, 717.5697632
1: -187.6941071, 711.3997803, -194.0026398, 735.4414673, -923.1355591, 905.4024048
2: -132.6405182, 736.5941772, -137.1115875, 762.2473145, -894.8878174, 873.7056885
3: -323.1858215, 623.7142334, -334.1413879, 644.3536987, -967.5394287, 957.8555298
4: -218.3145142, 631.2977905, -225.6249390, 652.4591675, -870.7736816, 856.9226685

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6810908, upper bound: 743.6815468
time: 0.67 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6808347, upper bound: 743.6808704
time: 0.67 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -113.3904114, 589.4231567, -108.5263824, 565.8104858, -679.2009277, 697.9495239
1: -185.3195343, 700.2048340, -177.1419830, 671.7973022, -857.1168213, 877.3468018
2: -130.8647766, 725.2958374, -125.1645660, 696.1809692, -827.0456543, 850.4603882
3: -319.0674438, 614.1246948, -305.3207092, 588.5136719, -907.5811157, 919.4454346
4: -215.3625641, 621.9359741, -205.9687653, 596.1086426, -811.4711914, 827.9047241

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6656990, upper bound: 743.6730206
time: 0.61 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6656990, upper bound: 743.6808377
time: 0.74 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -112.4720993, 584.7092896, -114.7271652, 598.8875122, -711.3596191, 699.4364624
1: -183.7790375, 694.5543823, -187.3312378, 711.1581421, -894.9371948, 881.8856201
2: -129.7962646, 719.5155029, -132.4292145, 736.5253296, -866.3215942, 851.9447021
3: -316.3895569, 609.0925903, -322.6647644, 623.0994263, -939.4888916, 931.7573242
4: -213.5906372, 616.8994751, -217.9647522, 630.8546753, -844.4452515, 834.8641357

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B1_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6799311, upper bound: 743.6809721
time: 0.65 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B1_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6799311, upper bound: 743.6809721
time: 0.62 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -113.3904114, 589.4231567, -112.2555389, 585.8367920, -699.2271729, 701.6786499
1: -185.3195343, 700.2048340, -183.2851715, 695.0819702, -880.4014893, 883.4899902
2: -130.8647766, 725.2958374, -129.5101318, 720.9764404, -851.8410645, 854.8059692
3: -319.0674438, 614.1246948, -316.0387573, 608.3713379, -927.4387207, 930.1633911
4: -215.3625641, 621.9359741, -213.0701752, 616.4915771, -831.8541260, 835.0061646

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6794594, upper bound: 743.6803775
time: 1.00 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6794594, upper bound: 743.6803775
time: 0.77 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -112.4720993, 584.7092896, -118.5923157, 619.6892700, -732.1613159, 703.3016357
1: -183.7790375, 694.5543823, -193.7226410, 735.4439087, -919.2229614, 888.2769165
2: -129.7962646, 719.5155029, -136.9709625, 762.5639038, -892.3601685, 856.4864502
3: -316.3895569, 609.0925903, -333.8375854, 643.9866943, -960.3760986, 942.9300537
4: -213.5906372, 616.8994751, -225.3805084, 652.3953247, -865.9859009, 842.2799683

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B2_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6797694, upper bound: 743.6805269
time: 0.73 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2_B2_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6797694, upper bound: 743.6805269
time: 1.39 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -104.3936157, 542.4821167, -110.9822235, 578.0343018, -682.4277954, 653.4643555
1: -170.4239044, 643.9945068, -181.2656708, 686.5562134, -856.9801025, 825.2601929
2: -120.3708038, 667.9304810, -128.0746155, 711.2026367, -831.5734253, 796.0050659
3: -293.7593689, 564.1147461, -312.0987244, 601.6737061, -895.4331055, 876.2135010
4: -198.0750427, 571.9409790, -210.7660065, 609.2480469, -807.3230591, 782.7069702

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6799250, upper bound: 743.6790155
time: 0.76 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6799250, upper bound: 743.6790155
time: 0.67 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -110.9781723, 578.6544189, -106.6564865, 555.8721313, -666.8502808, 685.3108521
1: -181.4703522, 685.9696655, -174.1715851, 660.0601807, -841.5305176, 860.1412354
2: -128.1993103, 712.9481812, -123.0532837, 684.0206909, -812.2199707, 836.0014648
3: -313.4414062, 599.9348145, -299.8086243, 577.9774780, -891.4187622, 899.7434082
4: -210.8901215, 609.1958618, -202.4515533, 585.5199585, -796.4100342, 811.6473999

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6800803, upper bound: 743.6790155
time: 0.63 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6800803, upper bound: 743.6790155
time: 0.70 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -115.5850449, 602.0866699, -104.5376282, 544.0339355, -659.6188965, 706.6242065
1: -188.8125610, 715.0158081, -170.6537323, 645.9414673, -834.7540283, 885.6695557
2: -133.4127655, 740.6810913, -120.5315857, 669.5638428, -802.9765625, 861.2126465
3: -325.2523499, 626.8279419, -293.9199524, 566.0307007, -891.2828979, 920.7479248
4: -219.5975342, 634.8402100, -198.2964783, 573.3177490, -792.9152832, 833.1367188

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6662976, upper bound: 743.6661526
time: 0.65 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1_A2_B1_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6662976, upper bound: 743.6799367
time: 0.56 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -115.5850449, 602.0866699, -110.3126907, 574.7211304, -690.3061523, 712.3993530
1: -188.8125610, 715.0158081, -180.1349792, 682.5983276, -871.4108887, 895.1507568
2: -133.4127655, 740.6810913, -127.2944565, 707.1036987, -840.5164795, 867.9755249
3: -325.2523499, 626.8279419, -310.1027222, 598.2090454, -923.4613647, 936.9306641
4: -219.5975342, 634.8402100, -209.4601288, 605.7213135, -825.3188477, 844.3003540

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6795384, upper bound: 743.6792954
time: 0.64 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1_A2_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6795892, upper bound: 743.6789845
time: 0.66 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -105.8579788, 550.8231201, -107.4538422, 560.6994629, -666.5574341, 658.2768555
1: -172.9937286, 654.0793457, -175.7576904, 665.4393921, -838.4329834, 829.8369751
2: -122.1078720, 677.7775269, -124.1703644, 690.0281372, -812.1359253, 801.9478760
3: -298.0981750, 573.3033447, -302.6199951, 582.5158081, -880.6139526, 875.9232788
4: -200.9350586, 580.8945923, -204.2857666, 590.2531738, -791.1881714, 785.1803589

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A1_B1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6800665, upper bound: 743.6791657
time: 0.61 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A1_B1_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6800665, upper bound: 743.6791657
time: 0.87 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -108.5724106, 564.6914673, -113.8203506, 594.0222168, -702.5945435, 678.5118408
1: -177.2688599, 670.5733643, -186.0420227, 705.0423584, -882.3111572, 856.6152954
2: -125.2067795, 694.9168091, -131.4514008, 730.9847412, -856.1914673, 826.3682251
3: -305.5324097, 587.8476562, -320.3628845, 617.4293213, -922.9615479, 908.2104492
4: -206.0745544, 595.5639648, -216.2687683, 625.4304810, -831.5050049, 811.8327637

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A1_B2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6755346, upper bound: 743.6739237
time: 0.62 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A1_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782814, upper bound: 743.6764188
time: 0.62 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A1_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782814, upper bound: 743.6764188
time: 0.90 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -115.5850449, 602.0866699, -107.9195786, 562.7070312, -678.2919922, 710.0062256
1: -188.8125610, 715.0158081, -176.3300476, 667.7414551, -856.5540161, 891.3457031
2: -133.4127655, 740.6810913, -124.5254211, 692.5445557, -825.9572754, 865.2065430
3: -325.2523499, 626.8279419, -303.8122864, 584.3259888, -909.5782471, 930.6401978
4: -219.5975342, 634.8402100, -204.8101654, 592.3217773, -811.9193115, 839.6503296

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A2_B1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6756608, upper bound: 743.6746030
time: 0.62 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A2_B1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802156, upper bound: 743.6795238
time: 0.62 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A2_B1_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6769109, upper bound: 743.6754143
time: 0.66 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -115.5850449, 602.0866699, -113.7788315, 593.9063110, -709.4912109, 715.8654785
1: -188.8125610, 715.0158081, -185.9134064, 704.8239746, -893.6365356, 900.9291382
2: -133.4127655, 740.6810913, -131.4050903, 730.8419800, -864.2546387, 872.0861816
3: -325.2523499, 626.8279419, -320.1673584, 617.1762085, -942.4285278, 946.9953003
4: -219.5975342, 634.8402100, -216.1784058, 625.2183228, -844.8158569, 851.0186157

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A2_B2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6756608, upper bound: 743.6746030
time: 0.69 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A2_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6805338, upper bound: 743.6793945
time: 0.82 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2_A2_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6769109, upper bound: 743.6794284
time: 0.64 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -118.0681763, 616.2675781, -109.1163712, 568.7373657, -686.8055420, 725.3839111
1: -192.8032684, 731.9175415, -178.0852661, 675.3342896, -868.1375732, 910.0028076
2: -136.2920685, 757.8210449, -125.8366547, 699.7525635, -836.0446167, 883.6577148
3: -332.1836243, 641.4924927, -306.9655457, 591.7061157, -923.8896484, 948.4580078
4: -224.3690338, 649.3289795, -207.0895386, 599.2673950, -823.6364136, 856.4184570

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1_B1_A2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6510323, upper bound: 743.6801468
time: 0.62 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1_B1_A2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6573400, upper bound: 743.6797582
time: 0.63 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -110.2316360, 573.4473267, -115.2489243, 601.5029297, -711.7345581, 688.6962280
1: -179.9872742, 680.9829102, -188.1741638, 714.2922974, -894.2795410, 869.1571045
2: -127.1333618, 705.6771240, -133.0243378, 739.7500610, -866.8833618, 838.7014160
3: -310.2410278, 597.0095825, -324.1338501, 625.9076538, -936.1486816, 921.1434326
4: -209.2506409, 604.7850952, -218.9506683, 633.6780396, -842.9286499, 823.7357178

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1_B2_A1_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6795353, upper bound: 743.6793433
time: 0.71 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1_B2_A1_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6795760, upper bound: 743.6791327
time: 0.65 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -115.5850449, 602.0866699, -115.2489243, 601.5029297, -717.0879517, 717.3355713
1: -188.8125610, 715.0158081, -188.1741638, 714.2922974, -903.1048584, 903.1899414
2: -133.4127655, 740.6810913, -133.0243378, 739.7500610, -873.1627808, 873.7053833
3: -325.2523499, 626.8279419, -324.1338501, 625.9076538, -951.1599121, 950.9617920
4: -219.5975342, 634.8402100, -218.9506683, 633.6780396, -853.2755737, 853.7907715

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1_B2_A2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6795353, upper bound: 743.6794754
time: 0.59 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1_B2_A2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6795760, upper bound: 743.6791389
time: 0.61 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -110.2316360, 573.4473267, -112.8706055, 588.8263550, -699.0579834, 686.3179321
1: -179.9872742, 680.9829102, -184.2608795, 698.7004395, -878.6876831, 865.2437134
2: -127.1333618, 705.6771240, -130.2121124, 724.6369019, -851.7701416, 835.8892212
3: -310.2410278, 597.0095825, -317.7404175, 611.6794434, -921.9204712, 914.7500000
4: -209.2506409, 604.7850952, -214.2491302, 619.7327271, -828.9833984, 819.0340576

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6745990, upper bound: 743.6739466
time: 0.69 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6791610, upper bound: 743.6797718
time: 0.64 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6801178, upper bound: 743.6797710
time: 0.62 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -115.5850449, 602.0866699, -112.8706055, 588.8263550, -704.4113770, 714.9572754
1: -188.8125610, 715.0158081, -184.2608795, 698.7004395, -887.5130005, 899.2766724
2: -133.4127655, 740.6810913, -130.2121124, 724.6369019, -858.0496216, 870.8931885
3: -325.2523499, 626.8279419, -317.7404175, 611.6794434, -936.9317627, 944.5682983
4: -219.5975342, 634.8402100, -214.2491302, 619.7327271, -839.3302612, 849.0891724

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6745990, upper bound: 743.6739800
time: 0.88 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6791610, upper bound: 743.6797876
time: 0.66 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B1_A2_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6801178, upper bound: 743.6797876
time: 0.90 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -110.2316360, 573.4473267, -119.2862473, 623.0567017, -733.2883301, 692.7335815
1: -179.9872742, 680.9829102, -194.8324585, 739.5347290, -919.5219116, 875.8153076
2: -127.1333618, 705.6771240, -137.7608185, 766.6956177, -893.8289185, 843.4379272
3: -310.2410278, 597.0095825, -335.7662354, 647.6977539, -957.9386597, 932.7758179
4: -209.2506409, 604.7850952, -226.7017059, 656.0482178, -865.2988281, 831.4868164

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6745990, upper bound: 743.6741143
time: 0.62 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_B1_A1_A1_B2_A2_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -115.5850449, 602.0866699, -119.2862473, 623.0567017, -738.6416626, 721.3729248
1: -188.8125610, 715.0158081, -194.8324585, 739.5347290, -928.3472900, 909.8482056
2: -133.4127655, 740.6810913, -137.7608185, 766.6956177, -900.1083984, 878.4418945
3: -325.2523499, 626.8279419, -335.7662354, 647.6977539, -972.9499512, 962.5941772
4: -219.5975342, 634.8402100, -226.7017059, 656.0482178, -875.6457520, 861.5419312

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6745990, upper bound: 743.6742015
time: 0.57 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.58 + 186.51 = 189.08 seconds
