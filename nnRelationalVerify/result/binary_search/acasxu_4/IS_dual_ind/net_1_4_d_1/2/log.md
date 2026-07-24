## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_4.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 0.055158916499999995


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479)
1: (-0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910)
2: (-0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850)
3: (-0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681)
4: (-0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295)

## BASE Result
execution time: IAR + LP analysis = 1.91 + 0.96 = 2.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0562259, upper bound: 0.0562259


# Binary Search by BASE starts (time budget: 1197.13 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1000000, mid=0.1000000, abs_max=0.058847926557064056
rel_dist={0: [-0.05599888835187894, 0.05599888835187895]}

## Binary search (step 1) starts
Candidate diff: 0.0500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0500000, mid=0.0500000, abs_max=0.058847926557064056
rel_dist={0: [-0.05565336822157154, 0.055653368221571534]}

## Binary search (step 2) starts
Candidate diff: 0.0250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0250000, mid=0.0250000, abs_max=0.058847926557064056
rel_dist={0: [-0.05542113047992016, 0.055421130479920144]}

## Binary search (step 3) starts
Candidate diff: 0.0125000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0125000, mid=0.0125000, abs_max=0.058847926557064056
rel_dist={0: [-0.05528714416572913, 0.05528714416572915]}

## Binary search (step 4) starts
Candidate diff: 0.0062500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0062500, mid=0.0062500, abs_max=0.058847926557064056
rel_dist={0: [-0.055212698399098425, 0.055212698398983934]}

## Binary search (step 5) starts
Candidate diff: 0.0031250


## IAR start
Binary search (step 5): status=Status.VERIFIED, low=0.0031250, high=0.0062500, mid=0.0031250, abs_max=0.058847926557064056
rel_dist={0: [-0.055132520784768234, 0.05513252078476821]}

## Binary search (step 6) starts
Candidate diff: 0.0046875


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0031250, high=0.0046875, mid=0.0046875, abs_max=0.058847926557064056
rel_dist={0: [-0.055185278748120487, 0.05518527874783165]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0031250, high=0.0039062, mid=0.0039062, abs_max=0.058847926557064056
rel_dist={0: [-0.055165241381868235, 0.05516524138172432]}

## Binary search (step 8) starts
Candidate diff: 0.0035156


## IAR start
Binary search (step 8): status=Status.VERIFIED, low=0.0035156, high=0.0039062, mid=0.0035156, abs_max=0.058847926557064056
rel_dist={0: [-0.05515501399134123, 0.055155013991219096]}

## Binary search (step 9) starts
Candidate diff: 0.0037109


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0035156, high=0.0037109, mid=0.0037109, abs_max=0.058847926557064056
rel_dist={0: [-0.05516014090469245, 0.055160140904564114]}

## Binary search (step 10) starts
Candidate diff: 0.0036133


## IAR start
Binary search (step 10): status=Status.VERIFIED, low=0.0036133, high=0.0037109, mid=0.0036133, abs_max=0.058847926557064056
rel_dist={0: [-0.05515757745100269, 0.05515757745087749]}

## Binary search (step 11) starts
Candidate diff: 0.0036621


## IAR start
Binary search (step 11): status=Status.VERIFIED, low=0.0036621, high=0.0037109, mid=0.0036621, abs_max=0.058847926557064056
rel_dist={0: [-0.05515885917960294, 0.05515885917960295]}

## Binary search (step 12) starts
Candidate diff: 0.0036865


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0036621, high=0.0036865, mid=0.0036865, abs_max=0.058847926557064056
rel_dist={0: [-0.05515950004240834, 0.05515950004228079]}

## Binary search (step 13) starts
Candidate diff: 0.0036743


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0036621, high=0.0036743, mid=0.0036743, abs_max=0.058847926557064056
rel_dist={0: [-0.0551591796108497, 0.05515917961084968]}

## Binary search (step 14) starts
Candidate diff: 0.0036682


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0036621, high=0.0036682, mid=0.0036682, abs_max=0.058847926557064056
rel_dist={0: [-0.05515901939457894, 0.05515901939432502]}

## Binary search (step 15) starts
Candidate diff: 0.0036652


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0036621, high=0.0036652, mid=0.0036652, abs_max=0.058847926557064056
rel_dist={0: [-0.05515893928599323, 0.05515893928586636]}

## Binary search (step 16) starts
Candidate diff: 0.0036636


## IAR start
Binary search (step 16): status=Status.VERIFIED, low=0.0036636, high=0.0036652, mid=0.0036636, abs_max=0.058847926557064056
rel_dist={0: [-0.05515889923307673, 0.05515889923294992]}

## Binary search (step 17) starts
Candidate diff: 0.0036644


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0036636, high=0.0036644, mid=0.0036644, abs_max=0.058847926557064056
rel_dist={0: [-0.05515891925993985, 0.05515891925968616]}

## Binary Search Result
Binary search time: 50.45 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.003663635035536572


# Individual Split (IS_dual_ind) starts
Time budget: 1146.68 seconds

## Binary search (step 0) starts
Candidate diff: 0.1018318


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556653
time: 0.35 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556482, upper bound: 0.0556482
time: 0.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.86 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556653
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -0.0556482, upper bound: 0.0556482

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0206600, 0.0216224, -0.0278593, 0.0309887, -0.0516487, 0.0494817
1: -0.0226827, 0.0442280, -0.0350513, 0.0705397, -0.0932224, 0.0792793
2: -0.0535189, 0.0294451, -0.0677722, 0.0423128, -0.0958317, 0.0972173
3: -0.0368305, 0.0571968, -0.0527389, 0.0981292, -0.1349597, 0.1099357
4: -0.0685483, 0.0351860, -0.0944105, 0.0499190, -0.1184673, 0.1295964

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555648, upper bound: 0.0555648
time: 0.37 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555648, upper bound: 0.0556482
time: 0.35 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0213640, 0.0238672, -0.0275995, 0.0305874, -0.0519514, 0.0514667
1: -0.0256137, 0.0480500, -0.0343877, 0.0693975, -0.0950112, 0.0824377
2: -0.0530172, 0.0317309, -0.0671558, 0.0413970, -0.0944141, 0.0988867
3: -0.0398649, 0.0629934, -0.0515191, 0.0963091, -0.1361740, 0.1145125
4: -0.0719933, 0.0355104, -0.0931684, 0.0484524, -0.1204456, 0.1286788

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556482, upper bound: 0.0555648
time: 0.39 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556482, upper bound: 0.0556482
time: 0.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.88 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -0.0555648, upper bound: 0.0555648
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -0.0555648, upper bound: 0.0556482
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -0.0556482, upper bound: 0.0555648
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -0.0556482, upper bound: 0.0556482

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0206600, 0.0216224, -0.0206600, 0.0216224, -0.0422824, 0.0422824
1: -0.0226827, 0.0442280, -0.0226827, 0.0442280, -0.0669107, 0.0669107
2: -0.0535189, 0.0294451, -0.0535189, 0.0294451, -0.0829640, 0.0829640
3: -0.0368305, 0.0571968, -0.0368305, 0.0571968, -0.0940273, 0.0940273
4: -0.0685483, 0.0351860, -0.0685483, 0.0351860, -0.1037342, 0.1037342

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556400
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556270
time: 0.37 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0206600, 0.0216224, -0.0213640, 0.0238672, -0.0445272, 0.0429864
1: -0.0226827, 0.0442280, -0.0256137, 0.0480500, -0.0707327, 0.0698417
2: -0.0535189, 0.0294451, -0.0530172, 0.0317309, -0.0852498, 0.0824623
3: -0.0368305, 0.0571968, -0.0398649, 0.0629934, -0.0998239, 0.0970616
4: -0.0685483, 0.0351860, -0.0719933, 0.0355104, -0.1040587, 0.1071792

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556578
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556638
time: 0.36 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0213640, 0.0238672, -0.0206600, 0.0216224, -0.0429864, 0.0445272
1: -0.0256137, 0.0480500, -0.0226827, 0.0442280, -0.0698417, 0.0707327
2: -0.0530172, 0.0317309, -0.0535189, 0.0294451, -0.0824623, 0.0852498
3: -0.0398649, 0.0629934, -0.0368305, 0.0571968, -0.0970616, 0.0998239
4: -0.0719933, 0.0355104, -0.0685483, 0.0351860, -0.1071792, 0.1040587

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555630, upper bound: 0.0550900
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556482, upper bound: 0.0555648
time: 0.37 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0213640, 0.0238672, -0.0213640, 0.0238672, -0.0452312, 0.0452312
1: -0.0256137, 0.0480500, -0.0256137, 0.0480500, -0.0736638, 0.0736638
2: -0.0530172, 0.0317309, -0.0530172, 0.0317309, -0.0847481, 0.0847481
3: -0.0398649, 0.0629934, -0.0398649, 0.0629934, -0.1028583, 0.1028583
4: -0.0719933, 0.0355104, -0.0719933, 0.0355104, -0.1075037, 0.1075037

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555630, upper bound: 0.0550900
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556482, upper bound: 0.0555648
time: 0.40 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.92 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556400
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556270
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556578
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556638
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -0.0555630, upper bound: 0.0550900
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -0.0556482, upper bound: 0.0555648
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -0.0555630, upper bound: 0.0550900
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -0.0556482, upper bound: 0.0555648

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0206600, 0.0216224, -0.0390627, 0.0390481
1: -0.0179365, 0.0346897, -0.0226827, 0.0442280, -0.0621645, 0.0573724
2: -0.0468226, 0.0235523, -0.0535189, 0.0294451, -0.0762677, 0.0770712
3: -0.0312331, 0.0437208, -0.0368305, 0.0571968, -0.0884299, 0.0805513
4: -0.0597628, 0.0287614, -0.0685483, 0.0351860, -0.0949488, 0.0973097

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556280, upper bound: 0.0556280
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556280, upper bound: 0.0556280
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0206258, 0.0215764, -0.0493751, 0.0522413
1: -0.0329395, 0.0745442, -0.0226243, 0.0440746, -0.0770141, 0.0971685
2: -0.0723793, 0.0529296, -0.0534147, 0.0293607, -0.1017400, 0.1063443
3: -0.0475005, 0.0975785, -0.0367476, 0.0569787, -0.1044792, 0.1343261
4: -0.0997654, 0.0589750, -0.0683751, 0.0350883, -0.1348538, 0.1273501

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556280, upper bound: 0.0556280
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556280, upper bound: 0.0556280
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0213640, 0.0238672, -0.0413075, 0.0397520
1: -0.0179365, 0.0346897, -0.0256137, 0.0480500, -0.0659865, 0.0603034
2: -0.0468226, 0.0235523, -0.0530172, 0.0317309, -0.0785535, 0.0765695
3: -0.0312331, 0.0437208, -0.0398649, 0.0629934, -0.0942266, 0.0835857
4: -0.0597628, 0.0287614, -0.0719933, 0.0355104, -0.0952732, 0.1007547

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556254
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556578
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0213278, 0.0238178, -0.0516166, 0.0529433
1: -0.0329395, 0.0745442, -0.0255377, 0.0478916, -0.0808311, 0.1000819
2: -0.0723793, 0.0529296, -0.0529076, 0.0316422, -0.1040216, 0.1058372
3: -0.0475005, 0.0975785, -0.0397803, 0.0627622, -0.1102628, 0.1373588
4: -0.0997654, 0.0589750, -0.0718101, 0.0354106, -0.1351760, 0.1307851

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556254
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556638
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0206600, 0.0216224, -0.0409574, 0.0422815
1: -0.0221682, 0.0413110, -0.0226827, 0.0442280, -0.0663962, 0.0639937
2: -0.0484735, 0.0279975, -0.0535189, 0.0294451, -0.0779186, 0.0815165
3: -0.0360067, 0.0530757, -0.0368305, 0.0571968, -0.0932035, 0.0899062
4: -0.0654610, 0.0313325, -0.0685483, 0.0351860, -0.1006470, 0.0998808

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556254, upper bound: 0.0550900
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556254, upper bound: 0.0550900
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0206258, 0.0215764, -0.0465290, 0.0495699
1: -0.0302124, 0.0647534, -0.0226243, 0.0440746, -0.0742869, 0.0873777
2: -0.0640738, 0.0455330, -0.0534147, 0.0293607, -0.0934345, 0.0989477
3: -0.0429836, 0.0844141, -0.0367476, 0.0569787, -0.0999623, 0.1211617
4: -0.0914269, 0.0490248, -0.0683751, 0.0350883, -0.1265153, 0.1173999

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556578, upper bound: 0.0555700
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556578, upper bound: 0.0555700
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0213640, 0.0238672, -0.0432022, 0.0429854
1: -0.0221682, 0.0413110, -0.0256137, 0.0480500, -0.0702183, 0.0669248
2: -0.0484735, 0.0279975, -0.0530172, 0.0317309, -0.0802044, 0.0810147
3: -0.0360067, 0.0530757, -0.0398649, 0.0629934, -0.0990001, 0.0929406
4: -0.0654610, 0.0313325, -0.0719933, 0.0355104, -0.1009714, 0.1033258

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0213278, 0.0238178, -0.0487704, 0.0502720
1: -0.0302124, 0.0647534, -0.0255377, 0.0478916, -0.0781039, 0.0902911
2: -0.0640738, 0.0455330, -0.0529076, 0.0316422, -0.0957161, 0.0984406
3: -0.0429836, 0.0844141, -0.0397803, 0.0627622, -0.1057459, 0.1241944
4: -0.0914269, 0.0490248, -0.0718101, 0.0354106, -0.1268375, 0.1208349

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555630
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555648
time: 0.39 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.75 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0556280, upper bound: 0.0556280
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0556280, upper bound: 0.0556280
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0556280, upper bound: 0.0556280
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0556280, upper bound: 0.0556280
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556254
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556578
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556254
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556638
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0556254, upper bound: 0.0550900
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0556254, upper bound: 0.0550900
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0556578, upper bound: 0.0555700
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0556578, upper bound: 0.0555700
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555630
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555648

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0174403, 0.0183881, -0.0358283, 0.0358283
1: -0.0179365, 0.0346897, -0.0179365, 0.0346897, -0.0526262, 0.0526262
2: -0.0468226, 0.0235523, -0.0468226, 0.0235523, -0.0703749, 0.0703749
3: -0.0312331, 0.0437208, -0.0312331, 0.0437208, -0.0749540, 0.0749540
4: -0.0597628, 0.0287614, -0.0597628, 0.0287614, -0.0885242, 0.0885242

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0556210
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555724, upper bound: 0.0555646
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0277988, 0.0316155, -0.0490558, 0.0461869
1: -0.0179365, 0.0346897, -0.0329395, 0.0745442, -0.0924807, 0.0676292
2: -0.0468226, 0.0235523, -0.0723793, 0.0529296, -0.0997523, 0.0959316
3: -0.0312331, 0.0437208, -0.0475005, 0.0975785, -0.1288116, 0.0912214
4: -0.0597628, 0.0287614, -0.0997654, 0.0589750, -0.1187378, 0.1285269

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0556210
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555724, upper bound: 0.0555646
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0174403, 0.0183881, -0.0461869, 0.0490558
1: -0.0329395, 0.0745442, -0.0179365, 0.0346897, -0.0676292, 0.0924807
2: -0.0723793, 0.0529296, -0.0468226, 0.0235523, -0.0959316, 0.0997523
3: -0.0475005, 0.0975785, -0.0312331, 0.0437208, -0.0912214, 0.1288116
4: -0.0997654, 0.0589750, -0.0597628, 0.0287614, -0.1285269, 0.1187378

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0556119
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555536, upper bound: 0.0555536
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0277988, 0.0316155, -0.0594143, 0.0594143
1: -0.0329395, 0.0745442, -0.0329395, 0.0745442, -0.1074837, 0.1074837
2: -0.0723793, 0.0529296, -0.0723793, 0.0529296, -0.1253090, 0.1253090
3: -0.0475005, 0.0975785, -0.0475005, 0.0975785, -0.1450790, 0.1450790
4: -0.0997654, 0.0589750, -0.0997654, 0.0589750, -0.1587404, 0.1587404

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0556119
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555536, upper bound: 0.0555536
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0193350, 0.0216215, -0.0390617, 0.0377230
1: -0.0179365, 0.0346897, -0.0221682, 0.0413110, -0.0592475, 0.0568580
2: -0.0468226, 0.0235523, -0.0484735, 0.0279975, -0.0748202, 0.0720258
3: -0.0312331, 0.0437208, -0.0360067, 0.0530757, -0.0843088, 0.0797275
4: -0.0597628, 0.0287614, -0.0654610, 0.0313325, -0.0910953, 0.0942224

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547159, upper bound: 0.0556188
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0556048
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0249527, 0.0289441, -0.0463844, 0.0433407
1: -0.0179365, 0.0346897, -0.0302124, 0.0647534, -0.0826899, 0.0649021
2: -0.0468226, 0.0235523, -0.0640738, 0.0455330, -0.0923556, 0.0876261
3: -0.0312331, 0.0437208, -0.0429836, 0.0844141, -0.1156472, 0.0867045
4: -0.0597628, 0.0287614, -0.0914269, 0.0490248, -0.1087876, 0.1201884

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547159, upper bound: 0.0556424
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0556264
time: 0.31 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0193350, 0.0216215, -0.0494203, 0.0509505
1: -0.0329395, 0.0745442, -0.0221682, 0.0413110, -0.0742506, 0.0967125
2: -0.0723793, 0.0529296, -0.0484735, 0.0279975, -0.1003769, 0.1014031
3: -0.0475005, 0.0975785, -0.0360067, 0.0530757, -0.1005763, 0.1335852
4: -0.0997654, 0.0589750, -0.0654610, 0.0313325, -0.1310980, 0.1244360

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547159, upper bound: 0.0556090
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0555936
time: 0.30 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0249527, 0.0289441, -0.0567429, 0.0565682
1: -0.0329395, 0.0745442, -0.0302124, 0.0647534, -0.0976930, 0.1047566
2: -0.0723793, 0.0529296, -0.0640738, 0.0455330, -0.1179123, 0.1170035
3: -0.0475005, 0.0975785, -0.0429836, 0.0844141, -0.1319146, 0.1405621
4: -0.0997654, 0.0589750, -0.0914269, 0.0490248, -0.1487902, 0.1504019

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547159, upper bound: 0.0556337
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0555981
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0174403, 0.0183881, -0.0377230, 0.0390617
1: -0.0221682, 0.0413110, -0.0179365, 0.0346897, -0.0568580, 0.0592475
2: -0.0484735, 0.0279975, -0.0468226, 0.0235523, -0.0720258, 0.0748202
3: -0.0360067, 0.0530757, -0.0312331, 0.0437208, -0.0797275, 0.0843088
4: -0.0654610, 0.0313325, -0.0597628, 0.0287614, -0.0942224, 0.0910953

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 7

Time for candidate selection: 2.69 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549335, upper bound: 0.0534249
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553988, upper bound: 0.0548294
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0277988, 0.0316155, -0.0509505, 0.0494203
1: -0.0221682, 0.0413110, -0.0329395, 0.0745442, -0.0967125, 0.0742506
2: -0.0484735, 0.0279975, -0.0723793, 0.0529296, -0.1014031, 0.1003769
3: -0.0360067, 0.0530757, -0.0475005, 0.0975785, -0.1335852, 0.1005763
4: -0.0654610, 0.0313325, -0.0997654, 0.0589750, -0.1244360, 0.1310980

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 7

Time for candidate selection: 2.75 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549335, upper bound: 0.0534249
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553988, upper bound: 0.0548294
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0174403, 0.0183881, -0.0433407, 0.0463844
1: -0.0302124, 0.0647534, -0.0179365, 0.0346897, -0.0649021, 0.0826899
2: -0.0640738, 0.0455330, -0.0468226, 0.0235523, -0.0876261, 0.0923556
3: -0.0429836, 0.0844141, -0.0312331, 0.0437208, -0.0867045, 0.1156472
4: -0.0914269, 0.0490248, -0.0597628, 0.0287614, -0.1201884, 0.1087876

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553767, upper bound: 0.0555700
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556264, upper bound: 0.0555002
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0277988, 0.0316155, -0.0565682, 0.0567429
1: -0.0302124, 0.0647534, -0.0329395, 0.0745442, -0.1047566, 0.0976930
2: -0.0640738, 0.0455330, -0.0723793, 0.0529296, -0.1170035, 0.1179123
3: -0.0429836, 0.0844141, -0.0475005, 0.0975785, -0.1405621, 0.1319146
4: -0.0914269, 0.0490248, -0.0997654, 0.0589750, -0.1504019, 0.1487902

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553767, upper bound: 0.0555700
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556264, upper bound: 0.0555002
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0193350, 0.0216215, -0.0409564, 0.0409564
1: -0.0221682, 0.0413110, -0.0221682, 0.0413110, -0.0634793, 0.0634793
2: -0.0484735, 0.0279975, -0.0484735, 0.0279975, -0.0764710, 0.0764710
3: -0.0360067, 0.0530757, -0.0360067, 0.0530757, -0.0890824, 0.0890824
4: -0.0654610, 0.0313325, -0.0654610, 0.0313325, -0.0967935, 0.0967935

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 7

Time for candidate selection: 2.74 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546916, upper bound: 0.0534249
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550479, upper bound: 0.0548294
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0249527, 0.0289441, -0.0482791, 0.0465741
1: -0.0221682, 0.0413110, -0.0302124, 0.0647534, -0.0869217, 0.0715234
2: -0.0484735, 0.0279975, -0.0640738, 0.0455330, -0.0940065, 0.0920714
3: -0.0360067, 0.0530757, -0.0429836, 0.0844141, -0.1204208, 0.0960593
4: -0.0654610, 0.0313325, -0.0914269, 0.0490248, -0.1144858, 0.1227595

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 7

Time for candidate selection: 2.77 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546916, upper bound: 0.0534249
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550479, upper bound: 0.0548294
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0193350, 0.0216215, -0.0465741, 0.0482791
1: -0.0302124, 0.0647534, -0.0221682, 0.0413110, -0.0715234, 0.0869217
2: -0.0640738, 0.0455330, -0.0484735, 0.0279975, -0.0920714, 0.0940065
3: -0.0429836, 0.0844141, -0.0360067, 0.0530757, -0.0960593, 0.1204208
4: -0.0914269, 0.0490248, -0.0654610, 0.0313325, -0.1227595, 0.1144858

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549628, upper bound: 0.0555630
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555064
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0249527, 0.0289441, -0.0538968, 0.0538968
1: -0.0302124, 0.0647534, -0.0302124, 0.0647534, -0.0949658, 0.0949658
2: -0.0640738, 0.0455330, -0.0640738, 0.0455330, -0.1096068, 0.1096068
3: -0.0429836, 0.0844141, -0.0429836, 0.0844141, -0.1273977, 0.1273977
4: -0.0914269, 0.0490248, -0.0914269, 0.0490248, -0.1404517, 0.1404517

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549628, upper bound: 0.0555648
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555064
time: 0.35 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.71 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0556210
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0555724, upper bound: 0.0555646
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0556210
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0555724, upper bound: 0.0555646
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0556119
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0555536, upper bound: 0.0555536
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0556119
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0555536, upper bound: 0.0555536
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0547159, upper bound: 0.0556188
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0556048
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0547159, upper bound: 0.0556424
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0556264
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0547159, upper bound: 0.0556090
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0555936
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0547159, upper bound: 0.0556337
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0555981
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0549335, upper bound: 0.0534249
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0553988, upper bound: 0.0548294
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0549335, upper bound: 0.0534249
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0553988, upper bound: 0.0548294
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0553767, upper bound: 0.0555700
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0556264, upper bound: 0.0555002
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0553767, upper bound: 0.0555700
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0556264, upper bound: 0.0555002
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0546916, upper bound: 0.0534249
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0550479, upper bound: 0.0548294
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0546916, upper bound: 0.0534249
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0550479, upper bound: 0.0548294
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0549628, upper bound: 0.0555630
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555064
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0549628, upper bound: 0.0555648
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555064

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0174403, 0.0183881, -0.0358727, 0.0359235
1: -0.0177545, 0.0346891, -0.0179365, 0.0346897, -0.0524442, 0.0526256
2: -0.0469266, 0.0241803, -0.0468226, 0.0235523, -0.0704789, 0.0710029
3: -0.0308810, 0.0437489, -0.0312331, 0.0437208, -0.0746018, 0.0749820
4: -0.0604094, 0.0290925, -0.0597628, 0.0287614, -0.0891708, 0.0888553

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552061, upper bound: 0.0552061
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552061, upper bound: 0.0555871
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0174403, 0.0183881, -0.0353017, 0.0352645
1: -0.0172633, 0.0331967, -0.0179365, 0.0346897, -0.0519530, 0.0511332
2: -0.0456765, 0.0224354, -0.0468226, 0.0235523, -0.0692288, 0.0692580
3: -0.0304100, 0.0416375, -0.0312331, 0.0437208, -0.0741309, 0.0728707
4: -0.0582723, 0.0274961, -0.0597628, 0.0287614, -0.0870337, 0.0872589

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555871, upper bound: 0.0552061
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555871, upper bound: 0.0555871
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0277988, 0.0316155, -0.0491002, 0.0462820
1: -0.0177545, 0.0346891, -0.0329395, 0.0745442, -0.0922987, 0.0676286
2: -0.0469266, 0.0241803, -0.0723793, 0.0529296, -0.0998563, 0.0965596
3: -0.0308810, 0.0437489, -0.0475005, 0.0975785, -0.1284595, 0.0912494
4: -0.0604094, 0.0290925, -0.0997654, 0.0589750, -0.1193844, 0.1288579

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0552061
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0555646
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0277988, 0.0316155, -0.0485292, 0.0456231
1: -0.0172633, 0.0331967, -0.0329395, 0.0745442, -0.0918075, 0.0661362
2: -0.0456765, 0.0224354, -0.0723793, 0.0529296, -0.0986062, 0.0948147
3: -0.0304100, 0.0416375, -0.0475005, 0.0975785, -0.1279885, 0.0891381
4: -0.0582723, 0.0274961, -0.0997654, 0.0589750, -0.1172473, 0.1272616

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555724, upper bound: 0.0552061
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555724, upper bound: 0.0555646
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0174403, 0.0183881, -0.0463969, 0.0498877
1: -0.0325964, 0.0775581, -0.0179365, 0.0346897, -0.0672861, 0.0954946
2: -0.0725141, 0.0540359, -0.0468226, 0.0235523, -0.0960664, 0.1008585
3: -0.0464773, 0.1009725, -0.0312331, 0.0437208, -0.0901981, 0.1322056
4: -0.1008744, 0.0594039, -0.0597628, 0.0287614, -0.1296358, 0.1191667

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552061, upper bound: 0.0551828
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552061, upper bound: 0.0555724
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0174403, 0.0183881, -0.0457259, 0.0485709
1: -0.0321509, 0.0732026, -0.0179365, 0.0346897, -0.0668406, 0.0911391
2: -0.0714296, 0.0520608, -0.0468226, 0.0235523, -0.0949819, 0.0988834
3: -0.0465936, 0.0955957, -0.0312331, 0.0437208, -0.0903144, 0.1268288
4: -0.0984768, 0.0579753, -0.0597628, 0.0287614, -0.1272383, 0.1177381

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555646, upper bound: 0.0551828
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555646, upper bound: 0.0555724
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0277988, 0.0316155, -0.0596244, 0.0602462
1: -0.0325964, 0.0775581, -0.0329395, 0.0745442, -0.1071406, 0.1104977
2: -0.0725141, 0.0540359, -0.0723793, 0.0529296, -0.1254437, 0.1264152
3: -0.0464773, 0.1009725, -0.0475005, 0.0975785, -0.1440558, 0.1484731
4: -0.1008744, 0.0594039, -0.0997654, 0.0589750, -0.1598494, 0.1591693

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0551828
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0555536
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0277988, 0.0316155, -0.0589534, 0.0589294
1: -0.0321509, 0.0732026, -0.0329395, 0.0745442, -0.1066951, 0.1061421
2: -0.0714296, 0.0520608, -0.0723793, 0.0529296, -0.1243593, 0.1244401
3: -0.0465936, 0.0955957, -0.0475005, 0.0975785, -0.1441721, 0.1430962
4: -0.0984768, 0.0579753, -0.0997654, 0.0589750, -0.1574518, 0.1577407

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555536, upper bound: 0.0551828
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555536, upper bound: 0.0555536
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0193350, 0.0216215, -0.0391061, 0.0378182
1: -0.0177545, 0.0346891, -0.0221682, 0.0413110, -0.0590656, 0.0568573
2: -0.0469266, 0.0241803, -0.0484735, 0.0279975, -0.0749241, 0.0726538
3: -0.0308810, 0.0437489, -0.0360067, 0.0530757, -0.0839567, 0.0797556
4: -0.0604094, 0.0290925, -0.0654610, 0.0313325, -0.0917419, 0.0945535

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 7

Time for candidate selection: 2.70 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0527846, upper bound: 0.0544105
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544502, upper bound: 0.0553921
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0193350, 0.0216215, -0.0385351, 0.0371592
1: -0.0172633, 0.0331967, -0.0221682, 0.0413110, -0.0585744, 0.0553650
2: -0.0456765, 0.0224354, -0.0484735, 0.0279975, -0.0736740, 0.0709089
3: -0.0304100, 0.0416375, -0.0360067, 0.0530757, -0.0834857, 0.0776442
4: -0.0582723, 0.0274961, -0.0654610, 0.0313325, -0.0896048, 0.0929571

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 7

Time for candidate selection: 2.85 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535715, upper bound: 0.0549680
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549866, upper bound: 0.0553721
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0249527, 0.0289441, -0.0464288, 0.0434359
1: -0.0177545, 0.0346891, -0.0302124, 0.0647534, -0.0825080, 0.0649014
2: -0.0469266, 0.0241803, -0.0640738, 0.0455330, -0.0924596, 0.0882542
3: -0.0308810, 0.0437489, -0.0429836, 0.0844141, -0.1152951, 0.0867325
4: -0.0604094, 0.0290925, -0.0914269, 0.0490248, -0.1094342, 0.1205194

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551293, upper bound: 0.0553767
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551293, upper bound: 0.0556264
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0249527, 0.0289441, -0.0458578, 0.0427769
1: -0.0172633, 0.0331967, -0.0302124, 0.0647534, -0.0820168, 0.0634091
2: -0.0456765, 0.0224354, -0.0640738, 0.0455330, -0.0912095, 0.0865092
3: -0.0304100, 0.0416375, -0.0429836, 0.0844141, -0.1148241, 0.0846211
4: -0.0582723, 0.0274961, -0.0914269, 0.0490248, -0.1072971, 0.1189231

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555192, upper bound: 0.0553767
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555192, upper bound: 0.0556264
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0193350, 0.0216215, -0.0496303, 0.0517824
1: -0.0325964, 0.0775581, -0.0221682, 0.0413110, -0.0739075, 0.0997264
2: -0.0725141, 0.0540359, -0.0484735, 0.0279975, -0.1005116, 0.1025094
3: -0.0464773, 0.1009725, -0.0360067, 0.0530757, -0.0995530, 0.1369792
4: -0.1008744, 0.0594039, -0.0654610, 0.0313325, -0.1322069, 0.1248649

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 7

Time for candidate selection: 2.96 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0527827, upper bound: 0.0544149
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544502, upper bound: 0.0553828
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0193350, 0.0216215, -0.0489594, 0.0504656
1: -0.0321509, 0.0732026, -0.0221682, 0.0413110, -0.0734620, 0.0953708
2: -0.0714296, 0.0520608, -0.0484735, 0.0279975, -0.0994272, 0.1005343
3: -0.0465936, 0.0955957, -0.0360067, 0.0530757, -0.0996693, 0.1316024
4: -0.0984768, 0.0579753, -0.0654610, 0.0313325, -0.1298094, 0.1234363

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 7

Time for candidate selection: 2.84 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534143, upper bound: 0.0549250
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548294, upper bound: 0.0553613
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0249527, 0.0289441, -0.0569530, 0.0574001
1: -0.0325964, 0.0775581, -0.0302124, 0.0647534, -0.0973499, 0.1077705
2: -0.0725141, 0.0540359, -0.0640738, 0.0455330, -0.1180471, 0.1181097
3: -0.0464773, 0.1009725, -0.0429836, 0.0844141, -0.1308914, 0.1439561
4: -0.1008744, 0.0594039, -0.0914269, 0.0490248, -0.1498992, 0.1508308

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551274, upper bound: 0.0550725
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551274, upper bound: 0.0555981
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0249527, 0.0289441, -0.0562820, 0.0560833
1: -0.0321509, 0.0732026, -0.0302124, 0.0647534, -0.0969044, 0.1034149
2: -0.0714296, 0.0520608, -0.0640738, 0.0455330, -0.1169626, 0.1161346
3: -0.0465936, 0.0955957, -0.0429836, 0.0844141, -0.1310077, 0.1385793
4: -0.0984768, 0.0579753, -0.0914269, 0.0490248, -0.1475016, 0.1494022

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551913, upper bound: 0.0550725
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551913, upper bound: 0.0555981
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0163063, 0.0177085, -0.0174403, 0.0183881, -0.0346943, 0.0351487
1: -0.0184658, 0.0330927, -0.0179365, 0.0346897, -0.0531555, 0.0510292
2: -0.0422703, 0.0221831, -0.0468226, 0.0235523, -0.0658226, 0.0690057
3: -0.0314417, 0.0417711, -0.0312331, 0.0437208, -0.0751626, 0.0730042
4: -0.0568412, 0.0248886, -0.0597628, 0.0287614, -0.0856027, 0.0846514

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553921, upper bound: 0.0544502
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553721, upper bound: 0.0549866
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0163063, 0.0177085, -0.0277988, 0.0316155, -0.0479218, 0.0455072
1: -0.0184658, 0.0330927, -0.0329395, 0.0745442, -0.0930100, 0.0660322
2: -0.0422703, 0.0221831, -0.0723793, 0.0529296, -0.0952000, 0.0945624
3: -0.0314417, 0.0417711, -0.0475005, 0.0975785, -0.1290202, 0.0892716
4: -0.0568412, 0.0248886, -0.0997654, 0.0589750, -0.1158163, 0.1246540

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553828, upper bound: 0.0544502
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553613, upper bound: 0.0548294
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0174403, 0.0183881, -0.0441507, 0.0479868
1: -0.0317584, 0.0698348, -0.0179365, 0.0346897, -0.0664481, 0.0877713
2: -0.0655043, 0.0478268, -0.0468226, 0.0235523, -0.0890566, 0.0946494
3: -0.0441914, 0.0912354, -0.0312331, 0.0437208, -0.0879122, 0.1224686
4: -0.0945988, 0.0507568, -0.0597628, 0.0287614, -0.1233602, 0.1105196

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553767, upper bound: 0.0551293
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553767, upper bound: 0.0555192
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0174403, 0.0183881, -0.0428692, 0.0458364
1: -0.0294868, 0.0633977, -0.0179365, 0.0346897, -0.0641765, 0.0813342
2: -0.0631345, 0.0446851, -0.0468226, 0.0235523, -0.0866868, 0.0915077
3: -0.0421317, 0.0824423, -0.0312331, 0.0437208, -0.0858525, 0.1136754
4: -0.0901240, 0.0480672, -0.0597628, 0.0287614, -0.1188854, 0.1078300

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556264, upper bound: 0.0551293
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556264, upper bound: 0.0555192
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0277988, 0.0316155, -0.0573782, 0.0583453
1: -0.0317584, 0.0698348, -0.0329395, 0.0745442, -0.1063026, 0.1027744
2: -0.0655043, 0.0478268, -0.0723793, 0.0529296, -0.1184339, 0.1202061
3: -0.0441914, 0.0912354, -0.0475005, 0.0975785, -0.1417699, 0.1387360
4: -0.0945988, 0.0507568, -0.0997654, 0.0589750, -0.1535738, 0.1505222

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553784, upper bound: 0.0551293
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553784, upper bound: 0.0555002
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0277988, 0.0316155, -0.0560967, 0.0561949
1: -0.0294868, 0.0633977, -0.0329395, 0.0745442, -0.1040310, 0.0963372
2: -0.0631345, 0.0446851, -0.0723793, 0.0529296, -0.1160641, 0.1170644
3: -0.0421317, 0.0824423, -0.0475005, 0.0975785, -0.1397102, 0.1299428
4: -0.0901240, 0.0480672, -0.0997654, 0.0589750, -0.1490990, 0.1478326

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556303, upper bound: 0.0551293
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556303, upper bound: 0.0555002
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0193350, 0.0216215, -0.0473841, 0.0498815
1: -0.0317584, 0.0698348, -0.0221682, 0.0413110, -0.0730694, 0.0920031
2: -0.0655043, 0.0478268, -0.0484735, 0.0279975, -0.0935018, 0.0963003
3: -0.0441914, 0.0912354, -0.0360067, 0.0530757, -0.0972671, 0.1272421
4: -0.0945988, 0.0507568, -0.0654610, 0.0313325, -0.1259313, 0.1162178

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 7

Time for candidate selection: 2.83 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0526808, upper bound: 0.0543595
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544914, upper bound: 0.0553366
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0193350, 0.0216215, -0.0461026, 0.0477310
1: -0.0294868, 0.0633977, -0.0221682, 0.0413110, -0.0707978, 0.0855659
2: -0.0631345, 0.0446851, -0.0484735, 0.0279975, -0.0911320, 0.0931586
3: -0.0421317, 0.0824423, -0.0360067, 0.0530757, -0.0952074, 0.1184490
4: -0.0901240, 0.0480672, -0.0654610, 0.0313325, -0.1214565, 0.1135281

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 7

Time for candidate selection: 3.09 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536329, upper bound: 0.0548850
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550479, upper bound: 0.0552877
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0249527, 0.0289441, -0.0547068, 0.0554992
1: -0.0317584, 0.0698348, -0.0302124, 0.0647534, -0.0965118, 0.1000472
2: -0.0655043, 0.0478268, -0.0640738, 0.0455330, -0.1110373, 0.1119006
3: -0.0441914, 0.0912354, -0.0429836, 0.0844141, -0.1286055, 0.1342191
4: -0.0945988, 0.0507568, -0.0914269, 0.0490248, -0.1436236, 0.1421837

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553513, upper bound: 0.0551142
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553513, upper bound: 0.0555064
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0249527, 0.0289441, -0.0534253, 0.0533487
1: -0.0294868, 0.0633977, -0.0302124, 0.0647534, -0.0942402, 0.0936100
2: -0.0631345, 0.0446851, -0.0640738, 0.0455330, -0.1086674, 0.1087589
3: -0.0421317, 0.0824423, -0.0429836, 0.0844141, -0.1265458, 0.1254259
4: -0.0901240, 0.0480672, -0.0914269, 0.0490248, -0.1391487, 0.1394941

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556151, upper bound: 0.0551142
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556151, upper bound: 0.0555064
time: 0.39 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.11 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0552061, upper bound: 0.0552061
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0552061, upper bound: 0.0555871
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0555871, upper bound: 0.0552061
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0555871, upper bound: 0.0555871
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0552061
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0555646
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0555724, upper bound: 0.0552061
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0555724, upper bound: 0.0555646
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0552061, upper bound: 0.0551828
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0552061, upper bound: 0.0555724
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0555646, upper bound: 0.0551828
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0555646, upper bound: 0.0555724
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0551828
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0555536
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0555536, upper bound: 0.0551828
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0555536, upper bound: 0.0555536
IS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0527846, upper bound: 0.0544105
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0544502, upper bound: 0.0553921
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0535715, upper bound: 0.0549680
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0549866, upper bound: 0.0553721
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0551293, upper bound: 0.0553767
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0551293, upper bound: 0.0556264
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0555192, upper bound: 0.0553767
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0555192, upper bound: 0.0556264
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0527827, upper bound: 0.0544149
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0544502, upper bound: 0.0553828
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0534143, upper bound: 0.0549250
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0548294, upper bound: 0.0553613
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0551274, upper bound: 0.0550725
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0551274, upper bound: 0.0555981
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0551913, upper bound: 0.0550725
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0551913, upper bound: 0.0555981
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0553921, upper bound: 0.0544502
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0553721, upper bound: 0.0549866
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0553828, upper bound: 0.0544502
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0553613, upper bound: 0.0548294
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0553767, upper bound: 0.0551293
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0553767, upper bound: 0.0555192
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0556264, upper bound: 0.0551293
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0556264, upper bound: 0.0555192
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0553784, upper bound: 0.0551293
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0553784, upper bound: 0.0555002
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0556303, upper bound: 0.0551293
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0556303, upper bound: 0.0555002
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0526808, upper bound: 0.0543595
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0544914, upper bound: 0.0553366
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0536329, upper bound: 0.0548850
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0550479, upper bound: 0.0552877
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0553513, upper bound: 0.0551142
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0553513, upper bound: 0.0555064
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0556151, upper bound: 0.0551142
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.11
Output dim: 0, lower bound: -0.0556151, upper bound: 0.0555064

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0174846, 0.0184832, -0.0359678, 0.0359678
1: -0.0177545, 0.0346891, -0.0177545, 0.0346891, -0.0524436, 0.0524436
2: -0.0469266, 0.0241803, -0.0469266, 0.0241803, -0.0711069, 0.0711069
3: -0.0308810, 0.0437489, -0.0308810, 0.0437489, -0.0746299, 0.0746299
4: -0.0604094, 0.0290925, -0.0604094, 0.0290925, -0.0895019, 0.0895019

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0553629
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544429
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0169136, 0.0178243, -0.0353089, 0.0353969
1: -0.0177545, 0.0346891, -0.0172633, 0.0331967, -0.0509512, 0.0519524
2: -0.0469266, 0.0241803, -0.0456765, 0.0224354, -0.0693620, 0.0698568
3: -0.0308810, 0.0437489, -0.0304100, 0.0416375, -0.0725185, 0.0741589
4: -0.0604094, 0.0290925, -0.0582723, 0.0274961, -0.0879055, 0.0873647

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0556145
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0549104
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0174846, 0.0184832, -0.0353969, 0.0353089
1: -0.0172633, 0.0331967, -0.0177545, 0.0346891, -0.0519524, 0.0509512
2: -0.0456765, 0.0224354, -0.0469266, 0.0241803, -0.0698568, 0.0693620
3: -0.0304100, 0.0416375, -0.0308810, 0.0437489, -0.0741589, 0.0725185
4: -0.0582723, 0.0274961, -0.0604094, 0.0290925, -0.0873647, 0.0879055

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551666, upper bound: 0.0530334
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555704, upper bound: 0.0551896
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0169136, 0.0178243, -0.0347379, 0.0347379
1: -0.0172633, 0.0331967, -0.0172633, 0.0331967, -0.0504600, 0.0504600
2: -0.0456765, 0.0224354, -0.0456765, 0.0224354, -0.0681119, 0.0681119
3: -0.0304100, 0.0416375, -0.0304100, 0.0416375, -0.0720475, 0.0720475
4: -0.0582723, 0.0274961, -0.0582723, 0.0274961, -0.0857684, 0.0857684

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551666, upper bound: 0.0535517
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555704, upper bound: 0.0552465
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0280088, 0.0324474, -0.0499320, 0.0464920
1: -0.0177545, 0.0346891, -0.0325964, 0.0775581, -0.0953127, 0.0672855
2: -0.0469266, 0.0241803, -0.0725141, 0.0540359, -0.1009625, 0.0966944
3: -0.0308810, 0.0437489, -0.0464773, 0.1009725, -0.1318535, 0.0902262
4: -0.0604094, 0.0290925, -0.1008744, 0.0594039, -0.1198133, 0.1299669

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548878, upper bound: 0.0553892
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0544692
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0273379, 0.0311306, -0.0486153, 0.0458211
1: -0.0177545, 0.0346891, -0.0321509, 0.0732026, -0.0909571, 0.0668400
2: -0.0469266, 0.0241803, -0.0714296, 0.0520608, -0.0989874, 0.0956100
3: -0.0308810, 0.0437489, -0.0465936, 0.0955957, -0.1264767, 0.0903425
4: -0.0604094, 0.0290925, -0.0984768, 0.0579753, -0.1183847, 0.1275693

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548878, upper bound: 0.0556145
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0548454
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0280088, 0.0324474, -0.0493610, 0.0458331
1: -0.0172633, 0.0331967, -0.0325964, 0.0775581, -0.0948215, 0.0657931
2: -0.0456765, 0.0224354, -0.0725141, 0.0540359, -0.0997124, 0.0949495
3: -0.0304100, 0.0416375, -0.0464773, 0.1009725, -0.1313825, 0.0881148
4: -0.0582723, 0.0274961, -0.1008744, 0.0594039, -0.1176762, 0.1283706

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551519, upper bound: 0.0530334
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555521, upper bound: 0.0551896
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0273379, 0.0311306, -0.0480443, 0.0451621
1: -0.0172633, 0.0331967, -0.0321509, 0.0732026, -0.0904659, 0.0653476
2: -0.0456765, 0.0224354, -0.0714296, 0.0520608, -0.0977373, 0.0938650
3: -0.0304100, 0.0416375, -0.0465936, 0.0955957, -0.1260057, 0.0882311
4: -0.0582723, 0.0274961, -0.0984768, 0.0579753, -0.1162476, 0.1259730

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551519, upper bound: 0.0535107
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555521, upper bound: 0.0552465
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0174846, 0.0184832, -0.0464920, 0.0499320
1: -0.0325964, 0.0775581, -0.0177545, 0.0346891, -0.0672855, 0.0953127
2: -0.0725141, 0.0540359, -0.0469266, 0.0241803, -0.0966944, 0.1009625
3: -0.0464773, 0.1009725, -0.0308810, 0.0437489, -0.0902262, 0.1318535
4: -0.1008744, 0.0594039, -0.0604094, 0.0290925, -0.1299669, 0.1198133

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549527, upper bound: 0.0529986
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551896, upper bound: 0.0553602
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0169136, 0.0178243, -0.0458331, 0.0493610
1: -0.0325964, 0.0775581, -0.0172633, 0.0331967, -0.0657931, 0.0948215
2: -0.0725141, 0.0540359, -0.0456765, 0.0224354, -0.0949495, 0.0997124
3: -0.0464773, 0.1009725, -0.0304100, 0.0416375, -0.0881148, 0.1313825
4: -0.1008744, 0.0594039, -0.0582723, 0.0274961, -0.1283706, 0.1176762

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549527, upper bound: 0.0534798
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551896, upper bound: 0.0555981
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0174846, 0.0184832, -0.0458211, 0.0486153
1: -0.0321509, 0.0732026, -0.0177545, 0.0346891, -0.0668400, 0.0909571
2: -0.0714296, 0.0520608, -0.0469266, 0.0241803, -0.0956100, 0.0989874
3: -0.0465936, 0.0955957, -0.0308810, 0.0437489, -0.0903425, 0.1264767
4: -0.0984768, 0.0579753, -0.0604094, 0.0290925, -0.1275693, 0.1183847

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551910, upper bound: 0.0531099
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555382, upper bound: 0.0551583
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0169136, 0.0178243, -0.0451621, 0.0480443
1: -0.0321509, 0.0732026, -0.0172633, 0.0331967, -0.0653476, 0.0904659
2: -0.0714296, 0.0520608, -0.0456765, 0.0224354, -0.0938650, 0.0977373
3: -0.0465936, 0.0955957, -0.0304100, 0.0416375, -0.0882311, 0.1260057
4: -0.0984768, 0.0579753, -0.0582723, 0.0274961, -0.1259730, 0.1162476

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551666, upper bound: 0.0535649
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555382, upper bound: 0.0551583
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0280088, 0.0324474, -0.0604562, 0.0604562
1: -0.0325964, 0.0775581, -0.0325964, 0.0775581, -0.1101546, 0.1101546
2: -0.0725141, 0.0540359, -0.0725141, 0.0540359, -0.1265500, 0.1265500
3: -0.0464773, 0.1009725, -0.0464773, 0.1009725, -0.1474498, 0.1474498
4: -0.1008744, 0.0594039, -0.1008744, 0.0594039, -0.1602783, 0.1602783

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549217, upper bound: 0.0529986
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551583, upper bound: 0.0553602
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0273379, 0.0311306, -0.0591394, 0.0597853
1: -0.0325964, 0.0775581, -0.0321509, 0.0732026, -0.1057990, 0.1097091
2: -0.0725141, 0.0540359, -0.0714296, 0.0520608, -0.1245749, 0.1254655
3: -0.0464773, 0.1009725, -0.0465936, 0.0955957, -0.1420730, 0.1475661
4: -0.1008744, 0.0594039, -0.0984768, 0.0579753, -0.1588497, 0.1578807

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549217, upper bound: 0.0533617
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551583, upper bound: 0.0555981
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0280088, 0.0324474, -0.0597853, 0.0591394
1: -0.0321509, 0.0732026, -0.0325964, 0.0775581, -0.1097091, 0.1057990
2: -0.0714296, 0.0520608, -0.0725141, 0.0540359, -0.1254655, 0.1245749
3: -0.0465936, 0.0955957, -0.0464773, 0.1009725, -0.1475661, 0.1420730
4: -0.0984768, 0.0579753, -0.1008744, 0.0594039, -0.1578807, 0.1588497

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551911, upper bound: 0.0531099
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555214, upper bound: 0.0551583
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0273379, 0.0311306, -0.0584685, 0.0584685
1: -0.0321509, 0.0732026, -0.0321509, 0.0732026, -0.1053535, 0.1053535
2: -0.0714296, 0.0520608, -0.0714296, 0.0520608, -0.1234904, 0.1234904
3: -0.0465936, 0.0955957, -0.0465936, 0.0955957, -0.1421893, 0.1421893
4: -0.0984768, 0.0579753, -0.0984768, 0.0579753, -0.1564521, 0.1564521

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551911, upper bound: 0.0535312
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555214, upper bound: 0.0551583
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0163063, 0.0177085, -0.0351931, 0.0347895
1: -0.0177545, 0.0346891, -0.0184658, 0.0330927, -0.0508472, 0.0531549
2: -0.0469266, 0.0241803, -0.0422703, 0.0221831, -0.0691097, 0.0664506
3: -0.0308810, 0.0437489, -0.0314417, 0.0417711, -0.0726521, 0.0751906
4: -0.0604094, 0.0290925, -0.0568412, 0.0248886, -0.0852980, 0.0859337

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544022, upper bound: 0.0553828
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542345, upper bound: 0.0547285
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0163063, 0.0177085, -0.0346221, 0.0341305
1: -0.0172633, 0.0331967, -0.0184658, 0.0330927, -0.0503560, 0.0516625
2: -0.0456765, 0.0224354, -0.0422703, 0.0221831, -0.0678596, 0.0647057
3: -0.0304100, 0.0416375, -0.0314417, 0.0417711, -0.0721811, 0.0730792
4: -0.0582723, 0.0274961, -0.0568412, 0.0248886, -0.0831609, 0.0843374

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0539447, upper bound: 0.0534840
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549506, upper bound: 0.0553576
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0257626, 0.0305466, -0.0480312, 0.0442459
1: -0.0177545, 0.0346891, -0.0317584, 0.0698348, -0.0875894, 0.0664474
2: -0.0469266, 0.0241803, -0.0655043, 0.0478268, -0.0947534, 0.0896846
3: -0.0308810, 0.0437489, -0.0441914, 0.0912354, -0.1221164, 0.0879403
4: -0.0604094, 0.0290925, -0.0945988, 0.0507568, -0.1111662, 0.1236912

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549722, upper bound: 0.0554695
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549439, upper bound: 0.0547069
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0244811, 0.0283961, -0.0458807, 0.0429644
1: -0.0177545, 0.0346891, -0.0294868, 0.0633977, -0.0811522, 0.0641759
2: -0.0469266, 0.0241803, -0.0631345, 0.0446851, -0.0916117, 0.0873148
3: -0.0308810, 0.0437489, -0.0421317, 0.0824423, -0.1133233, 0.0858805
4: -0.0604094, 0.0290925, -0.0901240, 0.0480672, -0.1084765, 0.1192164

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549722, upper bound: 0.0554695
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 41

Time for candidate selection: 2.92 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545848, upper bound: 0.0542830
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548637, upper bound: 0.0554092
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0257626, 0.0305466, -0.0474602, 0.0435869
1: -0.0172633, 0.0331967, -0.0317584, 0.0698348, -0.0870982, 0.0649551
2: -0.0456765, 0.0224354, -0.0655043, 0.0478268, -0.0935033, 0.0879397
3: -0.0304100, 0.0416375, -0.0441914, 0.0912354, -0.1216454, 0.0858289
4: -0.0582723, 0.0274961, -0.0945988, 0.0507568, -0.1090291, 0.1220949

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550904, upper bound: 0.0532294
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554911, upper bound: 0.0553596
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0244811, 0.0283961, -0.0453097, 0.0423054
1: -0.0172633, 0.0331967, -0.0294868, 0.0633977, -0.0806610, 0.0626835
2: -0.0456765, 0.0224354, -0.0631345, 0.0446851, -0.0903616, 0.0855699
3: -0.0304100, 0.0416375, -0.0421317, 0.0824423, -0.1128523, 0.0837692
4: -0.0582723, 0.0274961, -0.0901240, 0.0480672, -0.1063394, 0.1176201

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550904, upper bound: 0.0536820
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554911, upper bound: 0.0553657
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0163063, 0.0177085, -0.0457173, 0.0487537
1: -0.0325964, 0.0775581, -0.0184658, 0.0330927, -0.0656891, 0.0960240
2: -0.0725141, 0.0540359, -0.0422703, 0.0221831, -0.0946972, 0.0963062
3: -0.0464773, 0.1009725, -0.0314417, 0.0417711, -0.0882484, 0.1324142
4: -0.1008744, 0.0594039, -0.0568412, 0.0248886, -0.1257630, 0.1162451

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544022, upper bound: 0.0553700
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543173, upper bound: 0.0550938
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0163063, 0.0177085, -0.0450463, 0.0474369
1: -0.0321509, 0.0732026, -0.0184658, 0.0330927, -0.0652436, 0.0916684
2: -0.0714296, 0.0520608, -0.0422703, 0.0221831, -0.0936127, 0.0943311
3: -0.0465936, 0.0955957, -0.0314417, 0.0417711, -0.0883646, 0.1270374
4: -0.0984768, 0.0579753, -0.0568412, 0.0248886, -0.1233654, 0.1148166

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0539676, upper bound: 0.0535045
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547938, upper bound: 0.0553451
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0244811, 0.0283961, -0.0564049, 0.0569285
1: -0.0325964, 0.0775581, -0.0294868, 0.0633977, -0.0959941, 0.1070449
2: -0.0725141, 0.0540359, -0.0631345, 0.0446851, -0.1171992, 0.1171703
3: -0.0464773, 0.1009725, -0.0421317, 0.0824423, -0.1289196, 0.1431042
4: -0.1008744, 0.0594039, -0.0901240, 0.0480672, -0.1489415, 0.1495278

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548612, upper bound: 0.0535417
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550957, upper bound: 0.0556163
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0257626, 0.0305466, -0.0578844, 0.0568933
1: -0.0321509, 0.0732026, -0.0317584, 0.0698348, -0.1019858, 0.1049610
2: -0.0714296, 0.0520608, -0.0655043, 0.0478268, -0.1192564, 0.1175651
3: -0.0465936, 0.0955957, -0.0441914, 0.0912354, -0.1378290, 0.1397871
4: -0.0984768, 0.0579753, -0.0945988, 0.0507568, -0.1492336, 0.1525741

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551025, upper bound: 0.0530775
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551410, upper bound: 0.0550480
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0244811, 0.0283961, -0.0557340, 0.0556118
1: -0.0321509, 0.0732026, -0.0294868, 0.0633977, -0.0955486, 0.1026894
2: -0.0714296, 0.0520608, -0.0631345, 0.0446851, -0.1161148, 0.1151953
3: -0.0465936, 0.0955957, -0.0421317, 0.0824423, -0.1290358, 0.1377274
4: -0.0984768, 0.0579753, -0.0901240, 0.0480672, -0.1465440, 0.1480993

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551025, upper bound: 0.0535919
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551410, upper bound: 0.0550480
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0163063, 0.0177085, -0.0174846, 0.0184832, -0.0347895, 0.0351931
1: -0.0184658, 0.0330927, -0.0177545, 0.0346891, -0.0531549, 0.0508472
2: -0.0422703, 0.0221831, -0.0469266, 0.0241803, -0.0664506, 0.0691097
3: -0.0314417, 0.0417711, -0.0308810, 0.0437489, -0.0751906, 0.0726521
4: -0.0568412, 0.0248886, -0.0604094, 0.0290925, -0.0859337, 0.0852980

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 7

Time for candidate selection: 3.15 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551346, upper bound: 0.0541247
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553909, upper bound: 0.0544366
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553909, upper bound: 0.0544335
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0163063, 0.0177085, -0.0169136, 0.0178243, -0.0341305, 0.0346221
1: -0.0184658, 0.0330927, -0.0172633, 0.0331967, -0.0516625, 0.0503560
2: -0.0422703, 0.0221831, -0.0456765, 0.0224354, -0.0647057, 0.0678596
3: -0.0314417, 0.0417711, -0.0304100, 0.0416375, -0.0730792, 0.0721811
4: -0.0568412, 0.0248886, -0.0582723, 0.0274961, -0.0843374, 0.0831609

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 7

Time for candidate selection: 3.19 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551234, upper bound: 0.0547550
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553644, upper bound: 0.0548674
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553644, upper bound: 0.0548643
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0163063, 0.0177085, -0.0280088, 0.0324474, -0.0487537, 0.0457173
1: -0.0184658, 0.0330927, -0.0325964, 0.0775581, -0.0960240, 0.0656891
2: -0.0422703, 0.0221831, -0.0725141, 0.0540359, -0.0963062, 0.0946972
3: -0.0314417, 0.0417711, -0.0464773, 0.1009725, -0.1324142, 0.0882484
4: -0.0568412, 0.0248886, -0.1008744, 0.0594039, -0.1162451, 0.1257630

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 7

Time for candidate selection: 3.16 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553828, upper bound: 0.0544502
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553825, upper bound: 0.0544366
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553825, upper bound: 0.0544335
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0163063, 0.0177085, -0.0273379, 0.0311306, -0.0474369, 0.0450463
1: -0.0184658, 0.0330927, -0.0321509, 0.0732026, -0.0916684, 0.0652436
2: -0.0422703, 0.0221831, -0.0714296, 0.0520608, -0.0943311, 0.0936127
3: -0.0314417, 0.0417711, -0.0465936, 0.0955957, -0.1270374, 0.0883646
4: -0.0568412, 0.0248886, -0.0984768, 0.0579753, -0.1148166, 0.1233654

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 7

Time for candidate selection: 3.22 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553613, upper bound: 0.0548294
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553563, upper bound: 0.0548158
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553563, upper bound: 0.0548127
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0174846, 0.0184832, -0.0442459, 0.0480312
1: -0.0317584, 0.0698348, -0.0177545, 0.0346891, -0.0664474, 0.0875894
2: -0.0655043, 0.0478268, -0.0469266, 0.0241803, -0.0896846, 0.0947534
3: -0.0441914, 0.0912354, -0.0308810, 0.0437489, -0.0879403, 0.1221164
4: -0.0945988, 0.0507568, -0.0604094, 0.0290925, -0.1236912, 0.1111662

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551607, upper bound: 0.0532055
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553596, upper bound: 0.0553135
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0169136, 0.0178243, -0.0435869, 0.0474602
1: -0.0317584, 0.0698348, -0.0172633, 0.0331967, -0.0649551, 0.0870982
2: -0.0655043, 0.0478268, -0.0456765, 0.0224354, -0.0879397, 0.0935033
3: -0.0441914, 0.0912354, -0.0304100, 0.0416375, -0.0858289, 0.1216454
4: -0.0945988, 0.0507568, -0.0582723, 0.0274961, -0.1220949, 0.1090291

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551607, upper bound: 0.0537123
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553596, upper bound: 0.0555514
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0174846, 0.0184832, -0.0429644, 0.0458807
1: -0.0294868, 0.0633977, -0.0177545, 0.0346891, -0.0641759, 0.0811522
2: -0.0631345, 0.0446851, -0.0469266, 0.0241803, -0.0873148, 0.0916117
3: -0.0421317, 0.0824423, -0.0308810, 0.0437489, -0.0858805, 0.1133233
4: -0.0901240, 0.0480672, -0.0604094, 0.0290925, -0.1192164, 0.1084765

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552022, upper bound: 0.0532055
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556055, upper bound: 0.0550977
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0169136, 0.0178243, -0.0423054, 0.0453097
1: -0.0294868, 0.0633977, -0.0172633, 0.0331967, -0.0626835, 0.0806610
2: -0.0631345, 0.0446851, -0.0456765, 0.0224354, -0.0855699, 0.0903616
3: -0.0421317, 0.0824423, -0.0304100, 0.0416375, -0.0837692, 0.1128523
4: -0.0901240, 0.0480672, -0.0582723, 0.0274961, -0.1176201, 0.1063394

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552022, upper bound: 0.0532055
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556055, upper bound: 0.0551017
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0280088, 0.0324474, -0.0582100, 0.0585554
1: -0.0317584, 0.0698348, -0.0325964, 0.0775581, -0.1093165, 0.1024313
2: -0.0655043, 0.0478268, -0.0725141, 0.0540359, -0.1195401, 0.1203409
3: -0.0441914, 0.0912354, -0.0464773, 0.1009725, -0.1451639, 0.1377127
4: -0.0945988, 0.0507568, -0.1008744, 0.0594039, -0.1540027, 0.1516312

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551701, upper bound: 0.0532418
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553596, upper bound: 0.0553135
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0273379, 0.0311306, -0.0568933, 0.0578844
1: -0.0317584, 0.0698348, -0.0321509, 0.0732026, -0.1049610, 0.1019858
2: -0.0655043, 0.0478268, -0.0714296, 0.0520608, -0.1175651, 0.1192564
3: -0.0441914, 0.0912354, -0.0465936, 0.0955957, -0.1397871, 0.1378290
4: -0.0945988, 0.0507568, -0.0984768, 0.0579753, -0.1525741, 0.1492336

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551701, upper bound: 0.0536380
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553596, upper bound: 0.0555514
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0280088, 0.0324474, -0.0569285, 0.0564049
1: -0.0294868, 0.0633977, -0.0325964, 0.0775581, -0.1070449, 0.0959941
2: -0.0631345, 0.0446851, -0.0725141, 0.0540359, -0.1171703, 0.1171992
3: -0.0421317, 0.0824423, -0.0464773, 0.1009725, -0.1431042, 0.1289196
4: -0.0901240, 0.0480672, -0.1008744, 0.0594039, -0.1495278, 0.1489415

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552273, upper bound: 0.0532418
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556056, upper bound: 0.0550977
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0273379, 0.0311306, -0.0556118, 0.0557340
1: -0.0294868, 0.0633977, -0.0321509, 0.0732026, -0.1026894, 0.0955486
2: -0.0631345, 0.0446851, -0.0714296, 0.0520608, -0.1151953, 0.1161148
3: -0.0421317, 0.0824423, -0.0465936, 0.0955957, -0.1377274, 0.1290358
4: -0.0901240, 0.0480672, -0.0984768, 0.0579753, -0.1480993, 0.1465440

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552273, upper bound: 0.0535743
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556056, upper bound: 0.0551017
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0163063, 0.0177085, -0.0434711, 0.0468528
1: -0.0317584, 0.0698348, -0.0184658, 0.0330927, -0.0648511, 0.0883006
2: -0.0655043, 0.0478268, -0.0422703, 0.0221831, -0.0876873, 0.0900971
3: -0.0441914, 0.0912354, -0.0314417, 0.0417711, -0.0859625, 0.1226771
4: -0.0945988, 0.0507568, -0.0568412, 0.0248886, -0.1194874, 0.1075981

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541268, upper bound: 0.0543863
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0542601, upper bound: 0.0551788
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0163063, 0.0177085, -0.0421896, 0.0447024
1: -0.0294868, 0.0633977, -0.0184658, 0.0330927, -0.0625795, 0.0818635
2: -0.0631345, 0.0446851, -0.0422703, 0.0221831, -0.0853175, 0.0869554
3: -0.0421317, 0.0824423, -0.0314417, 0.0417711, -0.0839027, 0.1138840
4: -0.0901240, 0.0480672, -0.0568412, 0.0248886, -0.1150126, 0.1049084

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0539324, upper bound: 0.0535485
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550111, upper bound: 0.0552632
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0257626, 0.0305466, -0.0563092, 0.0563092
1: -0.0317584, 0.0698348, -0.0317584, 0.0698348, -0.1015932, 0.1015932
2: -0.0655043, 0.0478268, -0.0655043, 0.0478268, -0.1133310, 0.1133310
3: -0.0441914, 0.0912354, -0.0441914, 0.0912354, -0.1354268, 0.1354268
4: -0.0945988, 0.0507568, -0.0945988, 0.0507568, -0.1453556, 0.1453556

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551667, upper bound: 0.0535113
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553269, upper bound: 0.0553110
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0244811, 0.0283961, -0.0541587, 0.0550277
1: -0.0317584, 0.0698348, -0.0294868, 0.0633977, -0.0951561, 0.0993216
2: -0.0655043, 0.0478268, -0.0631345, 0.0446851, -0.1101894, 0.1109612
3: -0.0441914, 0.0912354, -0.0421317, 0.0824423, -0.1266336, 0.1333671
4: -0.0945988, 0.0507568, -0.0901240, 0.0480672, -0.1426659, 0.1408808

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551667, upper bound: 0.0538178
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553269, upper bound: 0.0555455
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0257626, 0.0305466, -0.0550277, 0.0541587
1: -0.0294868, 0.0633977, -0.0317584, 0.0698348, -0.0993216, 0.0951561
2: -0.0631345, 0.0446851, -0.0655043, 0.0478268, -0.1109612, 0.1101894
3: -0.0421317, 0.0824423, -0.0441914, 0.0912354, -0.1333671, 0.1266336
4: -0.0901240, 0.0480672, -0.0945988, 0.0507568, -0.1408808, 0.1426659

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552273, upper bound: 0.0534960
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555827, upper bound: 0.0550850
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0244811, 0.0283961, -0.0528772, 0.0528772
1: -0.0294868, 0.0633977, -0.0294868, 0.0633977, -0.0928845, 0.0928845
2: -0.0631345, 0.0446851, -0.0631345, 0.0446851, -0.1078196, 0.1078196
3: -0.0421317, 0.0824423, -0.0421317, 0.0824423, -0.1245739, 0.1245739
4: -0.0901240, 0.0480672, -0.0901240, 0.0480672, -0.1381911, 0.1381911

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552273, upper bound: 0.0536437
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555827, upper bound: 0.0550850
time: 0.35 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.94 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0553629
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544429
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0556145
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0549104
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551666, upper bound: 0.0530334
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0555704, upper bound: 0.0551896
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551666, upper bound: 0.0535517
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0555704, upper bound: 0.0552465
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0548878, upper bound: 0.0553892
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0544692
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0548878, upper bound: 0.0556145
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0548454
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551519, upper bound: 0.0530334
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0555521, upper bound: 0.0551896
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551519, upper bound: 0.0535107
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0555521, upper bound: 0.0552465
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0549527, upper bound: 0.0529986
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551896, upper bound: 0.0553602
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0549527, upper bound: 0.0534798
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551896, upper bound: 0.0555981
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551910, upper bound: 0.0531099
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0555382, upper bound: 0.0551583
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551666, upper bound: 0.0535649
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0555382, upper bound: 0.0551583
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0549217, upper bound: 0.0529986
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551583, upper bound: 0.0553602
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0549217, upper bound: 0.0533617
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551583, upper bound: 0.0555981
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551911, upper bound: 0.0531099
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0555214, upper bound: 0.0551583
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551911, upper bound: 0.0535312
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0555214, upper bound: 0.0551583
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0544022, upper bound: 0.0553828
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0542345, upper bound: 0.0547285
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0539447, upper bound: 0.0534840
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0549506, upper bound: 0.0553576
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0549722, upper bound: 0.0554695
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0549439, upper bound: 0.0547069
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0545848, upper bound: 0.0542830
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0548637, upper bound: 0.0554092
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0550904, upper bound: 0.0532294
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0554911, upper bound: 0.0553596
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0550904, upper bound: 0.0536820
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0554911, upper bound: 0.0553657
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0544022, upper bound: 0.0553700
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0543173, upper bound: 0.0550938
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0539676, upper bound: 0.0535045
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0547938, upper bound: 0.0553451
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0548612, upper bound: 0.0535417
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0550957, upper bound: 0.0556163
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551025, upper bound: 0.0530775
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551410, upper bound: 0.0550480
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551025, upper bound: 0.0535919
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551410, upper bound: 0.0550480
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0553909, upper bound: 0.0544366
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0553909, upper bound: 0.0544335
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0553644, upper bound: 0.0548674
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0553644, upper bound: 0.0548643
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0553825, upper bound: 0.0544366
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0553825, upper bound: 0.0544335
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0553563, upper bound: 0.0548158
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0553563, upper bound: 0.0548127
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551607, upper bound: 0.0532055
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0553596, upper bound: 0.0553135
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551607, upper bound: 0.0537123
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0553596, upper bound: 0.0555514
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0552022, upper bound: 0.0532055
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0556055, upper bound: 0.0550977
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0552022, upper bound: 0.0532055
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0556055, upper bound: 0.0551017
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551701, upper bound: 0.0532418
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0553596, upper bound: 0.0553135
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551701, upper bound: 0.0536380
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0553596, upper bound: 0.0555514
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0552273, upper bound: 0.0532418
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0556056, upper bound: 0.0550977
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0552273, upper bound: 0.0535743
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0556056, upper bound: 0.0551017
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0541268, upper bound: 0.0543863
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0542601, upper bound: 0.0551788
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0539324, upper bound: 0.0535485
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0550111, upper bound: 0.0552632
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551667, upper bound: 0.0535113
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0553269, upper bound: 0.0553110
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0551667, upper bound: 0.0538178
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0553269, upper bound: 0.0555455
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0552273, upper bound: 0.0534960
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0555827, upper bound: 0.0550850
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0552273, upper bound: 0.0536437
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0555827, upper bound: 0.0550850

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0174754, 0.0184720, -0.0373048, 0.0365933
1: -0.0185345, 0.0336902, -0.0177350, 0.0346528, -0.0531874, 0.0514252
2: -0.0465600, 0.0245035, -0.0469067, 0.0241624, -0.0707224, 0.0714102
3: -0.0314490, 0.0414548, -0.0308536, 0.0437005, -0.0751495, 0.0723084
4: -0.0564701, 0.0289175, -0.0603823, 0.0290719, -0.0855421, 0.0892997

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544429
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544429
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0169040, 0.0178128, -0.0366457, 0.0360219
1: -0.0185345, 0.0336902, -0.0172409, 0.0331591, -0.0516936, 0.0509311
2: -0.0465600, 0.0245035, -0.0456564, 0.0224181, -0.0689781, 0.0701599
3: -0.0314490, 0.0414548, -0.0303782, 0.0415816, -0.0730306, 0.0718331
4: -0.0564701, 0.0289175, -0.0582444, 0.0274763, -0.0839464, 0.0871619

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529945, upper bound: 0.0551538
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549181, upper bound: 0.0556043
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0104994, 0.0109936, -0.0173591, 0.0182901, -0.0287894, 0.0283527
1: -0.0105939, 0.0073405, -0.0175916, 0.0340958, -0.0446898, 0.0249321
2: -0.0211140, 0.0063235, -0.0464559, 0.0237158, -0.0448298, 0.0527794
3: -0.0221017, 0.0017709, -0.0306682, 0.0428985, -0.0650002, 0.0324392
4: -0.0187051, 0.0056055, -0.0596313, 0.0285762, -0.0472813, 0.0652368

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551538, upper bound: 0.0529945
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544593, upper bound: 0.0529945
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0163001, 0.0171033, -0.0174846, 0.0184832, -0.0347834, 0.0345879
1: -0.0163531, 0.0312095, -0.0177545, 0.0346891, -0.0510422, 0.0489641
2: -0.0443946, 0.0211404, -0.0469266, 0.0241803, -0.0685749, 0.0680670
3: -0.0292436, 0.0388854, -0.0308810, 0.0437489, -0.0729924, 0.0697664
4: -0.0566717, 0.0260474, -0.0604094, 0.0290925, -0.0857642, 0.0864568

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556043, upper bound: 0.0549181
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547972, upper bound: 0.0548758
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0104994, 0.0109936, -0.0167804, 0.0176270, -0.0281264, 0.0277739
1: -0.0105939, 0.0073405, -0.0170922, 0.0325839, -0.0431778, 0.0244327
2: -0.0211140, 0.0063235, -0.0451919, 0.0219596, -0.0430736, 0.0515155
3: -0.0221017, 0.0017709, -0.0301882, 0.0407666, -0.0628683, 0.0319591
4: -0.0187051, 0.0056055, -0.0574700, 0.0269614, -0.0456665, 0.0630756

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533079, upper bound: 0.0533077
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533079, upper bound: 0.0535517
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0163001, 0.0171033, -0.0169136, 0.0178243, -0.0341244, 0.0340169
1: -0.0163531, 0.0312095, -0.0172633, 0.0331967, -0.0495499, 0.0484729
2: -0.0443946, 0.0211404, -0.0456765, 0.0224354, -0.0668300, 0.0668169
3: -0.0292436, 0.0388854, -0.0304100, 0.0416375, -0.0708811, 0.0692954
4: -0.0566717, 0.0260474, -0.0582723, 0.0274961, -0.0841679, 0.0843197

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536317, upper bound: 0.0549938
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0536317, upper bound: 0.0552465
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0280046, 0.0324436, -0.0512765, 0.0471225
1: -0.0185345, 0.0336902, -0.0325854, 0.0775451, -0.0960796, 0.0662755
2: -0.0465600, 0.0245035, -0.0725071, 0.0540298, -0.1005898, 0.0970106
3: -0.0314490, 0.0414548, -0.0464654, 0.1009535, -0.1324025, 0.0879202
4: -0.0564701, 0.0289175, -0.1008647, 0.0593972, -0.1158673, 0.1297822

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529373, upper bound: 0.0550854
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548459, upper bound: 0.0553710
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0273323, 0.0311268, -0.0499596, 0.0464502
1: -0.0185345, 0.0336902, -0.0321383, 0.0731863, -0.0917208, 0.0658284
2: -0.0465600, 0.0245035, -0.0714220, 0.0520547, -0.0986147, 0.0959255
3: -0.0314490, 0.0414548, -0.0465805, 0.0955694, -0.1270184, 0.0880353
4: -0.0564701, 0.0289175, -0.0984656, 0.0579685, -0.1144387, 0.1273831

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0530699, upper bound: 0.0551694
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549521, upper bound: 0.0556043
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0163001, 0.0171033, -0.0280088, 0.0324474, -0.0487475, 0.0451121
1: -0.0163531, 0.0312095, -0.0325964, 0.0775581, -0.0939113, 0.0638060
2: -0.0443946, 0.0211404, -0.0725141, 0.0540359, -0.0984305, 0.0936545
3: -0.0292436, 0.0388854, -0.0464773, 0.1009725, -0.1302161, 0.0853627
4: -0.0566717, 0.0260474, -0.1008744, 0.0594039, -0.1160756, 0.1269218

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534798, upper bound: 0.0549527
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0534798, upper bound: 0.0551896
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0163001, 0.0171033, -0.0273379, 0.0311306, -0.0474308, 0.0444412
1: -0.0163531, 0.0312095, -0.0321509, 0.0732026, -0.0895557, 0.0633605
2: -0.0443946, 0.0211404, -0.0714296, 0.0520608, -0.0964554, 0.0925701
3: -0.0292436, 0.0388854, -0.0465936, 0.0955957, -0.1248393, 0.0854789
4: -0.0566717, 0.0260474, -0.0984768, 0.0579753, -0.1146471, 0.1245242

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536523, upper bound: 0.0550107
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0536523, upper bound: 0.0552465
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0275401, 0.0318847, -0.0174846, 0.0184832, -0.0460233, 0.0493693
1: -0.0315660, 0.0759994, -0.0177545, 0.0346891, -0.0662550, 0.0937540
2: -0.0714210, 0.0530449, -0.0469266, 0.0241803, -0.0956013, 0.0999715
3: -0.0452617, 0.0986971, -0.0308810, 0.0437489, -0.0890106, 0.1295781
4: -0.0993945, 0.0582935, -0.0604094, 0.0290925, -0.1284870, 0.1187029

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553710, upper bound: 0.0548459
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543515, upper bound: 0.0548036
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0275401, 0.0318847, -0.0169136, 0.0178243, -0.0453644, 0.0487983
1: -0.0315660, 0.0759994, -0.0172633, 0.0331967, -0.0647627, 0.0932628
2: -0.0714210, 0.0530449, -0.0456765, 0.0224354, -0.0938564, 0.0987214
3: -0.0452617, 0.0986971, -0.0304100, 0.0416375, -0.0868993, 0.1291071
4: -0.0993945, 0.0582935, -0.0582723, 0.0274961, -0.1268907, 0.1165658

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0530334, upper bound: 0.0551519
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0530334, upper bound: 0.0555981
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0194715, 0.0185953, -0.0173591, 0.0182901, -0.0377616, 0.0359545
1: -0.0236661, 0.0364224, -0.0175916, 0.0340958, -0.0577619, 0.0540140
2: -0.0416546, 0.0217706, -0.0464559, 0.0237158, -0.0653703, 0.0682264
3: -0.0353903, 0.0438848, -0.0306682, 0.0428985, -0.0782888, 0.0745531
4: -0.0484835, 0.0241966, -0.0596313, 0.0285762, -0.0770597, 0.0838279

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551694, upper bound: 0.0530699
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544647, upper bound: 0.0530624
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0268415, 0.0305504, -0.0174846, 0.0184832, -0.0453247, 0.0480350
1: -0.0310880, 0.0714282, -0.0177545, 0.0346891, -0.0657771, 0.0891827
2: -0.0703224, 0.0510674, -0.0469266, 0.0241803, -0.0945028, 0.0979940
3: -0.0453250, 0.0927791, -0.0308810, 0.0437489, -0.0890739, 0.1236601
4: -0.0969780, 0.0568669, -0.0604094, 0.0290925, -0.1260705, 0.1172763

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556043, upper bound: 0.0549521
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547199, upper bound: 0.0549097
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0194715, 0.0185953, -0.0167804, 0.0176270, -0.0370985, 0.0353757
1: -0.0236661, 0.0364224, -0.0170922, 0.0325839, -0.0562500, 0.0535146
2: -0.0416546, 0.0217706, -0.0451919, 0.0219596, -0.0636141, 0.0669625
3: -0.0353903, 0.0438848, -0.0301882, 0.0407666, -0.0761569, 0.0740730
4: -0.0484835, 0.0241966, -0.0574700, 0.0269614, -0.0754449, 0.0816666

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533084, upper bound: 0.0533208
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533084, upper bound: 0.0533208
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0268415, 0.0305504, -0.0169136, 0.0178243, -0.0446657, 0.0474641
1: -0.0310880, 0.0714282, -0.0172633, 0.0331967, -0.0642847, 0.0886915
2: -0.0703224, 0.0510674, -0.0456765, 0.0224354, -0.0927578, 0.0967439
3: -0.0453250, 0.0927791, -0.0304100, 0.0416375, -0.0869625, 0.1231891
4: -0.0969780, 0.0568669, -0.0582723, 0.0274961, -0.1244742, 0.1151392

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534829, upper bound: 0.0548581
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534829, upper bound: 0.0548581
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0275401, 0.0318847, -0.0280088, 0.0324474, -0.0599875, 0.0598935
1: -0.0315660, 0.0759994, -0.0325964, 0.0775581, -0.1091241, 0.1085959
2: -0.0714210, 0.0530449, -0.0725141, 0.0540359, -0.1254568, 0.1255590
3: -0.0452617, 0.0986971, -0.0464773, 0.1009725, -0.1462342, 0.1451744
4: -0.0993945, 0.0582935, -0.1008744, 0.0594039, -0.1587984, 0.1591679

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529209, upper bound: 0.0550748
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0529209, upper bound: 0.0553602
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0275401, 0.0318847, -0.0273379, 0.0311306, -0.0586707, 0.0592225
1: -0.0315660, 0.0759994, -0.0321509, 0.0732026, -0.1047685, 0.1081504
2: -0.0714210, 0.0530449, -0.0714296, 0.0520608, -0.1234818, 0.1244745
3: -0.0452617, 0.0986971, -0.0465936, 0.0955957, -0.1408574, 0.1452906
4: -0.0993945, 0.0582935, -0.0984768, 0.0579753, -0.1573699, 0.1567703

Time for backsubstitution: 2.06 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0036636, high=0.1018318, mid=0.1018318, abs_max=0.058847926557064056
rel_dist={0: [-0.05600625688426092, 0.05600625688426092]}

## Binary search (step 1) starts
Candidate diff: 0.0527477


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0554034
time: 0.31 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553942, upper bound: 0.0553942
time: 0.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.77 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0554034
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -0.0553942, upper bound: 0.0553942

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0206600, 0.0216224, -0.0252327, 0.0271660, -0.0478260, 0.0468551
1: -0.0226827, 0.0442280, -0.0300615, 0.0599364, -0.0826191, 0.0742894
2: -0.0535189, 0.0294451, -0.0625250, 0.0373616, -0.0908805, 0.0919701
3: -0.0368305, 0.0571968, -0.0458409, 0.0809816, -0.1178121, 0.1030376
4: -0.0685483, 0.0351860, -0.0836424, 0.0439409, -0.1124892, 0.1188284

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553391, upper bound: 0.0553391
time: 0.31 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553391, upper bound: 0.0553942
time: 0.29 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0213640, 0.0238672, -0.0252271, 0.0273805, -0.0487444, 0.0490943
1: -0.0256137, 0.0480500, -0.0305537, 0.0594033, -0.0850170, 0.0786038
2: -0.0530172, 0.0317309, -0.0617325, 0.0368690, -0.0898862, 0.0934634
3: -0.0398649, 0.0629934, -0.0464436, 0.0803663, -0.1202312, 0.1094370
4: -0.0719933, 0.0355104, -0.0826624, 0.0428012, -0.1147945, 0.1181729

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552298, upper bound: 0.0553318
time: 0.32 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553942, upper bound: 0.0553942
time: 0.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.49 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -0.0553391, upper bound: 0.0553391
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -0.0553391, upper bound: 0.0553942
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -0.0552298, upper bound: 0.0553318
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -0.0553942, upper bound: 0.0553942

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0206600, 0.0216224, -0.0206600, 0.0216224, -0.0422824, 0.0422824
1: -0.0226827, 0.0442280, -0.0226827, 0.0442280, -0.0669107, 0.0669107
2: -0.0535189, 0.0294451, -0.0535189, 0.0294451, -0.0829640, 0.0829640
3: -0.0368305, 0.0571968, -0.0368305, 0.0571968, -0.0940273, 0.0940273
4: -0.0685483, 0.0351860, -0.0685483, 0.0351860, -0.1037342, 0.1037342

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0554018
time: 0.29 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0553994
time: 0.31 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0206600, 0.0216224, -0.0213640, 0.0238672, -0.0445272, 0.0429864
1: -0.0226827, 0.0442280, -0.0256137, 0.0480500, -0.0707327, 0.0698417
2: -0.0535189, 0.0294451, -0.0530172, 0.0317309, -0.0852498, 0.0824623
3: -0.0368305, 0.0571968, -0.0398649, 0.0629934, -0.0998239, 0.0970616
4: -0.0685483, 0.0351860, -0.0719933, 0.0355104, -0.1040587, 0.1071792

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0554029
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0554034
time: 0.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0206048, 0.0230341, -0.0216025, 0.0229951, -0.0435999, 0.0446365
1: -0.0241881, 0.0453783, -0.0242106, 0.0456234, -0.0698114, 0.0695889
2: -0.0513226, 0.0303424, -0.0539529, 0.0303841, -0.0817067, 0.0842953
3: -0.0383645, 0.0590674, -0.0388668, 0.0592243, -0.0975888, 0.0979342
4: -0.0694719, 0.0339743, -0.0695151, 0.0356756, -0.1051475, 0.1034894

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0552189
time: 0.31 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0553318
time: 0.32 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0207159, 0.0229785, -0.0367084, 0.0447420, -0.0654579, 0.0596868
1: -0.0242495, 0.0451704, -0.0515942, 0.1122316, -0.1364811, 0.0967646
2: -0.0509778, 0.0302205, -0.0882678, 0.0638482, -0.1148260, 0.1184883
3: -0.0382687, 0.0587770, -0.0728361, 0.1630409, -0.2013096, 0.1316131
4: -0.0686012, 0.0337570, -0.1400478, 0.0710807, -0.1396819, 0.1738049

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553318, upper bound: 0.0552298
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553318, upper bound: 0.0553942
time: 0.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.56 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0554018
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0553994
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0554029
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0554034
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0552189
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0553318
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0553318, upper bound: 0.0552298
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0553318, upper bound: 0.0553942

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0196564, 0.0205747, -0.0380150, 0.0380445
1: -0.0179365, 0.0346897, -0.0211653, 0.0410781, -0.0590146, 0.0558550
2: -0.0468226, 0.0235523, -0.0513527, 0.0274871, -0.0743097, 0.0749050
3: -0.0312331, 0.0437208, -0.0350417, 0.0527042, -0.0839373, 0.0787625
4: -0.0597628, 0.0287614, -0.0655577, 0.0330881, -0.0928509, 0.0943191

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554099, upper bound: 0.0554099
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554099, upper bound: 0.0554099
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0197872, 0.0204470, -0.0482458, 0.0514027
1: -0.0329395, 0.0745442, -0.0211767, 0.0403664, -0.0733060, 0.0957209
2: -0.0723793, 0.0529296, -0.0508505, 0.0273815, -0.0997609, 0.1037801
3: -0.0475005, 0.0975785, -0.0346786, 0.0517276, -0.0992281, 0.1322571
4: -0.0997654, 0.0589750, -0.0641330, 0.0327745, -0.1325400, 0.1231080

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554099, upper bound: 0.0554099
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554099, upper bound: 0.0554099
time: 0.29 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0206048, 0.0230341, -0.0404743, 0.0389928
1: -0.0179365, 0.0346897, -0.0241881, 0.0453783, -0.0633148, 0.0588778
2: -0.0468226, 0.0235523, -0.0513226, 0.0303424, -0.0771650, 0.0748749
3: -0.0312331, 0.0437208, -0.0383645, 0.0590674, -0.0903006, 0.0820854
4: -0.0597628, 0.0287614, -0.0694719, 0.0339743, -0.0937371, 0.0982333

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0553737
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0554029
time: 0.31 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0207159, 0.0229785, -0.0507772, 0.0523314
1: -0.0329395, 0.0745442, -0.0242495, 0.0451704, -0.0781099, 0.0987937
2: -0.0723793, 0.0529296, -0.0509778, 0.0302205, -0.1025998, 0.1039075
3: -0.0475005, 0.0975785, -0.0382687, 0.0587770, -0.1062776, 0.1358472
4: -0.0997654, 0.0589750, -0.0686012, 0.0337570, -0.1335225, 0.1275762

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0553737
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0554034
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0216025, 0.0229951, -0.0423301, 0.0432240
1: -0.0221682, 0.0413110, -0.0242106, 0.0456234, -0.0677916, 0.0655216
2: -0.0484735, 0.0279975, -0.0539529, 0.0303841, -0.0788576, 0.0819504
3: -0.0360067, 0.0530757, -0.0388668, 0.0592243, -0.0952310, 0.0919425
4: -0.0654610, 0.0313325, -0.0695151, 0.0356756, -0.1011366, 0.1008476

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0551679
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0551679
time: 0.30 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0216025, 0.0229951, -0.0479478, 0.0505466
1: -0.0302124, 0.0647534, -0.0242106, 0.0456234, -0.0758357, 0.0889640
2: -0.0640738, 0.0455330, -0.0539529, 0.0303841, -0.0944580, 0.0994859
3: -0.0429836, 0.0844141, -0.0388668, 0.0592243, -0.1022079, 0.1232809
4: -0.0914269, 0.0490248, -0.0695151, 0.0356756, -0.1271025, 0.1185399

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0553318
time: 0.30 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0553318
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0367084, 0.0447420, -0.0640769, 0.0583298
1: -0.0221682, 0.0413110, -0.0515942, 0.1122316, -0.1343998, 0.0929053
2: -0.0484735, 0.0279975, -0.0882678, 0.0638482, -0.1123217, 0.1162654
3: -0.0360067, 0.0530757, -0.0728361, 0.1630409, -0.1990476, 0.1259118
4: -0.0654610, 0.0313325, -0.1400478, 0.0710807, -0.1365417, 0.1713804

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0550899
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0550899
time: 0.31 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0367084, 0.0447420, -0.0696946, 0.0656525
1: -0.0302124, 0.0647534, -0.0515942, 0.1122316, -0.1424439, 0.1163477
2: -0.0640738, 0.0455330, -0.0882678, 0.0638482, -0.1279220, 0.1338008
3: -0.0429836, 0.0844141, -0.0728361, 0.1630409, -0.2060245, 0.1572502
4: -0.0914269, 0.0490248, -0.1400478, 0.0710807, -0.1625077, 0.1890726

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0553391
time: 0.30 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0553391
time: 0.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.60 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0554099, upper bound: 0.0554099
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0554099, upper bound: 0.0554099
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0554099, upper bound: 0.0554099
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0554099, upper bound: 0.0554099
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0553737
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0554029
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0553737
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0554034
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0551679
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0551679
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0553318
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0553318
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0550899
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0550899
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0553391
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0553391

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0174403, 0.0183881, -0.0358283, 0.0358283
1: -0.0179365, 0.0346897, -0.0179365, 0.0346897, -0.0526262, 0.0526262
2: -0.0468226, 0.0235523, -0.0468226, 0.0235523, -0.0703749, 0.0703749
3: -0.0312331, 0.0437208, -0.0312331, 0.0437208, -0.0749540, 0.0749540
4: -0.0597628, 0.0287614, -0.0597628, 0.0287614, -0.0885242, 0.0885242

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551756, upper bound: 0.0554091
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554024, upper bound: 0.0554126
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0276679, 0.0305874, -0.0480277, 0.0460560
1: -0.0179365, 0.0346897, -0.0324436, 0.0720466, -0.0899831, 0.0671333
2: -0.0468226, 0.0235523, -0.0715335, 0.0507885, -0.0976112, 0.0950858
3: -0.0312331, 0.0437208, -0.0468301, 0.0942273, -0.1254604, 0.0905509
4: -0.0597628, 0.0287614, -0.0975729, 0.0569583, -0.1167211, 0.1263343

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551756, upper bound: 0.0554091
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554024, upper bound: 0.0554126
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0173510, 0.0182791, -0.0460779, 0.0489665
1: -0.0329395, 0.0745442, -0.0178209, 0.0343542, -0.0672937, 0.0923651
2: -0.0723793, 0.0529296, -0.0466004, 0.0233044, -0.0956837, 0.0995301
3: -0.0475005, 0.0975785, -0.0310903, 0.0432695, -0.0907701, 0.1286689
4: -0.0997654, 0.0589750, -0.0593948, 0.0285069, -0.1282723, 0.1183698

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551756, upper bound: 0.0554025
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554024, upper bound: 0.0554024
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0276679, 0.0305874, -0.0583862, 0.0592835
1: -0.0329395, 0.0745442, -0.0324436, 0.0720466, -0.1049861, 0.1069878
2: -0.0723793, 0.0529296, -0.0715335, 0.0507885, -0.1231679, 0.1244631
3: -0.0475005, 0.0975785, -0.0468301, 0.0942273, -0.1417278, 0.1444086
4: -0.0997654, 0.0589750, -0.0975729, 0.0569583, -0.1567237, 0.1565479

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551756, upper bound: 0.0554025
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554024, upper bound: 0.0554024
time: 0.31 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0193350, 0.0216215, -0.0390617, 0.0377230
1: -0.0179365, 0.0346897, -0.0221682, 0.0413110, -0.0592475, 0.0568580
2: -0.0468226, 0.0235523, -0.0484735, 0.0279975, -0.0748202, 0.0720258
3: -0.0312331, 0.0437208, -0.0360067, 0.0530757, -0.0843088, 0.0797275
4: -0.0597628, 0.0287614, -0.0654610, 0.0313325, -0.0910953, 0.0942224

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546440, upper bound: 0.0549029
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551679, upper bound: 0.0553942
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0249527, 0.0289441, -0.0463844, 0.0433407
1: -0.0179365, 0.0346897, -0.0302124, 0.0647534, -0.0826899, 0.0649021
2: -0.0468226, 0.0235523, -0.0640738, 0.0455330, -0.0923556, 0.0876261
3: -0.0312331, 0.0437208, -0.0429836, 0.0844141, -0.1156472, 0.0867045
4: -0.0597628, 0.0287614, -0.0914269, 0.0490248, -0.1087876, 0.1201884

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546440, upper bound: 0.0554010
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551679, upper bound: 0.0554029
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0193350, 0.0216215, -0.0494203, 0.0509505
1: -0.0329395, 0.0745442, -0.0221682, 0.0413110, -0.0742506, 0.0967125
2: -0.0723793, 0.0529296, -0.0484735, 0.0279975, -0.1003769, 0.1014031
3: -0.0475005, 0.0975785, -0.0360067, 0.0530757, -0.1005763, 0.1335852
4: -0.0997654, 0.0589750, -0.0654610, 0.0313325, -0.1310980, 0.1244360

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546440, upper bound: 0.0549100
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0553737
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0249527, 0.0289441, -0.0567429, 0.0565682
1: -0.0329395, 0.0745442, -0.0302124, 0.0647534, -0.0976930, 0.1047566
2: -0.0723793, 0.0529296, -0.0640738, 0.0455330, -0.1179123, 0.1170035
3: -0.0475005, 0.0975785, -0.0429836, 0.0844141, -0.1319146, 0.1405621
4: -0.0997654, 0.0589750, -0.0914269, 0.0490248, -0.1487902, 0.1504019

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546440, upper bound: 0.0553823
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546440, upper bound: 0.0553823
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0173510, 0.0182791, -0.0376141, 0.0389725
1: -0.0221682, 0.0413110, -0.0178209, 0.0343542, -0.0565225, 0.0591319
2: -0.0484735, 0.0279975, -0.0466004, 0.0233044, -0.0717779, 0.0745979
3: -0.0360067, 0.0530757, -0.0310903, 0.0432695, -0.0792762, 0.0841661
4: -0.0654610, 0.0313325, -0.0593948, 0.0285069, -0.0939679, 0.0907274

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 7

Time for candidate selection: 2.63 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546251, upper bound: 0.0535822
time: 0.31 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549944, upper bound: 0.0549466
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0193350, 0.0216215, -0.0409564, 0.0409564
1: -0.0221682, 0.0413110, -0.0221682, 0.0413110, -0.0634793, 0.0634793
2: -0.0484735, 0.0279975, -0.0484735, 0.0279975, -0.0764710, 0.0764710
3: -0.0360067, 0.0530757, -0.0360067, 0.0530757, -0.0890824, 0.0890824
4: -0.0654610, 0.0313325, -0.0654610, 0.0313325, -0.0967935, 0.0967935

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 7

Time for candidate selection: 2.66 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546251, upper bound: 0.0535822
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549944, upper bound: 0.0549466
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0173510, 0.0182791, -0.0432318, 0.0462952
1: -0.0302124, 0.0647534, -0.0178209, 0.0343542, -0.0645666, 0.0825743
2: -0.0640738, 0.0455330, -0.0466004, 0.0233044, -0.0873782, 0.0921334
3: -0.0429836, 0.0844141, -0.0310903, 0.0432695, -0.0862531, 0.1155044
4: -0.0914269, 0.0490248, -0.0593948, 0.0285069, -0.1199338, 0.1084196

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547985, upper bound: 0.0548512
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552298, upper bound: 0.0553318
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0193350, 0.0216215, -0.0465741, 0.0482791
1: -0.0302124, 0.0647534, -0.0221682, 0.0413110, -0.0715234, 0.0869217
2: -0.0640738, 0.0455330, -0.0484735, 0.0279975, -0.0920714, 0.0940065
3: -0.0429836, 0.0844141, -0.0360067, 0.0530757, -0.0960593, 0.1204208
4: -0.0914269, 0.0490248, -0.0654610, 0.0313325, -0.1227595, 0.1144858

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547985, upper bound: 0.0548512
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552298, upper bound: 0.0553318
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0414462, 0.0518560, -0.0711910, 0.0630677
1: -0.0221682, 0.0413110, -0.0627760, 0.1324598, -0.1546281, 0.1040870
2: -0.0484735, 0.0279975, -0.0970791, 0.0711206, -0.1195941, 0.1250766
3: -0.0360067, 0.0530757, -0.0889659, 0.1968575, -0.2328642, 0.1420416
4: -0.0654610, 0.0313325, -0.1602575, 0.0791221, -0.1445831, 0.1915900

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 7

Time for candidate selection: 2.66 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546377, upper bound: 0.0534241
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549944, upper bound: 0.0548279
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0237000, 0.0276457, -0.0469807, 0.0453215
1: -0.0221682, 0.0413110, -0.0270187, 0.0608168, -0.0829850, 0.0683298
2: -0.0484735, 0.0279975, -0.0619147, 0.0439293, -0.0924028, 0.0899122
3: -0.0360067, 0.0530757, -0.0392698, 0.0784101, -0.1144168, 0.0923455
4: -0.0654610, 0.0313325, -0.0880449, 0.0474173, -0.1128783, 0.1193774

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 7

Time for candidate selection: 2.70 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546377, upper bound: 0.0534241
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551043, upper bound: 0.0548279
time: 0.32 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0414462, 0.0518560, -0.0768087, 0.0703904
1: -0.0302124, 0.0647534, -0.0627760, 0.1324598, -0.1626722, 0.1275294
2: -0.0640738, 0.0455330, -0.0970791, 0.0711206, -0.1351945, 0.1426120
3: -0.0429836, 0.0844141, -0.0889659, 0.1968575, -0.2398411, 0.1733800
4: -0.0914269, 0.0490248, -0.1602575, 0.0791221, -0.1705490, 0.2092823

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552474, upper bound: 0.0553391
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553937, upper bound: 0.0553391
time: 0.32 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0237000, 0.0276457, -0.0525984, 0.0526442
1: -0.0302124, 0.0647534, -0.0270187, 0.0608168, -0.0910291, 0.0917722
2: -0.0640738, 0.0455330, -0.0619147, 0.0439293, -0.1080031, 0.1074476
3: -0.0429836, 0.0844141, -0.0392698, 0.0784101, -0.1213937, 0.1236839
4: -0.0914269, 0.0490248, -0.0880449, 0.0474173, -0.1388442, 0.1370697

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552474, upper bound: 0.0553391
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553937, upper bound: 0.0553391
time: 0.32 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.70 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0551756, upper bound: 0.0554091
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0554024, upper bound: 0.0554126
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0551756, upper bound: 0.0554091
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0554024, upper bound: 0.0554126
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0551756, upper bound: 0.0554025
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0554024, upper bound: 0.0554024
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0551756, upper bound: 0.0554025
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0554024, upper bound: 0.0554024
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0546440, upper bound: 0.0549029
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0551679, upper bound: 0.0553942
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0546440, upper bound: 0.0554010
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0551679, upper bound: 0.0554029
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0546440, upper bound: 0.0549100
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0553737
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0546440, upper bound: 0.0553823
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0546440, upper bound: 0.0553823
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0546251, upper bound: 0.0535822
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0549944, upper bound: 0.0549466
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0546251, upper bound: 0.0535822
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0549944, upper bound: 0.0549466
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0547985, upper bound: 0.0548512
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0552298, upper bound: 0.0553318
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0547985, upper bound: 0.0548512
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0552298, upper bound: 0.0553318
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0546377, upper bound: 0.0534241
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0549944, upper bound: 0.0548279
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0546377, upper bound: 0.0534241
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0551043, upper bound: 0.0548279
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0552474, upper bound: 0.0553391
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0553937, upper bound: 0.0553391
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0552474, upper bound: 0.0553391
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -0.0553937, upper bound: 0.0553391

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0171597, 0.0180793, -0.0355639, 0.0356430
1: -0.0177545, 0.0346891, -0.0175864, 0.0337842, -0.0515387, 0.0522754
2: -0.0469266, 0.0241803, -0.0461216, 0.0228752, -0.0698018, 0.0703019
3: -0.0308810, 0.0437489, -0.0307810, 0.0424956, -0.0733766, 0.0745299
4: -0.0604094, 0.0290925, -0.0587734, 0.0279737, -0.0883831, 0.0878659

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552057, upper bound: 0.0552057
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552057, upper bound: 0.0554091
time: 0.30 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0174403, 0.0183881, -0.0353017, 0.0352645
1: -0.0172633, 0.0331967, -0.0179365, 0.0346897, -0.0519530, 0.0511332
2: -0.0456765, 0.0224354, -0.0468226, 0.0235523, -0.0692288, 0.0692580
3: -0.0304100, 0.0416375, -0.0312331, 0.0437208, -0.0741309, 0.0728707
4: -0.0582723, 0.0274961, -0.0597628, 0.0287614, -0.0870337, 0.0872589

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554091, upper bound: 0.0552057
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554091, upper bound: 0.0554129
time: 0.29 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0274396, 0.0303596, -0.0478443, 0.0459228
1: -0.0177545, 0.0346891, -0.0321042, 0.0713162, -0.0890708, 0.0667932
2: -0.0469266, 0.0241803, -0.0710153, 0.0503340, -0.0972606, 0.0951956
3: -0.0308810, 0.0437489, -0.0464226, 0.0932205, -0.1241015, 0.0901715
4: -0.0604094, 0.0290925, -0.0968829, 0.0563978, -0.1168072, 0.1259754

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551756, upper bound: 0.0552057
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551756, upper bound: 0.0554091
time: 0.30 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0276679, 0.0305874, -0.0475011, 0.0454922
1: -0.0172633, 0.0331967, -0.0324436, 0.0720466, -0.0893099, 0.0656403
2: -0.0456765, 0.0224354, -0.0715335, 0.0507885, -0.0964651, 0.0939689
3: -0.0304100, 0.0416375, -0.0468301, 0.0942273, -0.1246373, 0.0884676
4: -0.0582723, 0.0274961, -0.0975729, 0.0569583, -0.1152306, 0.1250690

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554022, upper bound: 0.0552057
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554022, upper bound: 0.0554126
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0171597, 0.0180793, -0.0460881, 0.0496071
1: -0.0325964, 0.0775581, -0.0175864, 0.0337842, -0.0663806, 0.0951445
2: -0.0725141, 0.0540359, -0.0461216, 0.0228752, -0.0953893, 0.1001575
3: -0.0464773, 0.1009725, -0.0307810, 0.0424956, -0.0889729, 0.1317535
4: -0.1008744, 0.0594039, -0.0587734, 0.0279737, -0.1288481, 0.1181773

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552057, upper bound: 0.0551756
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552057, upper bound: 0.0554022
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0173510, 0.0182791, -0.0456170, 0.0484816
1: -0.0321509, 0.0732026, -0.0178209, 0.0343542, -0.0665051, 0.0910234
2: -0.0714296, 0.0520608, -0.0466004, 0.0233044, -0.0947340, 0.0986612
3: -0.0465936, 0.0955957, -0.0310903, 0.0432695, -0.0898631, 0.1266861
4: -0.0984768, 0.0579753, -0.0593948, 0.0285069, -0.1269837, 0.1173701

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554091, upper bound: 0.0551756
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554091, upper bound: 0.0551756
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0274396, 0.0303596, -0.0583685, 0.0598869
1: -0.0325964, 0.0775581, -0.0321042, 0.0713162, -0.1039127, 0.1096623
2: -0.0725141, 0.0540359, -0.0710153, 0.0503340, -0.1228481, 0.1250512
3: -0.0464773, 0.1009725, -0.0464226, 0.0932205, -0.1396978, 0.1473951
4: -0.1008744, 0.0594039, -0.0968829, 0.0563978, -0.1572722, 0.1562868

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551756, upper bound: 0.0551756
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551756, upper bound: 0.0554022
time: 0.30 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0276679, 0.0305874, -0.0579253, 0.0587985
1: -0.0321509, 0.0732026, -0.0324436, 0.0720466, -0.1041975, 0.1056461
2: -0.0714296, 0.0520608, -0.0715335, 0.0507885, -0.1222182, 0.1235943
3: -0.0465936, 0.0955957, -0.0468301, 0.0942273, -0.1408209, 0.1424258
4: -0.0984768, 0.0579753, -0.0975729, 0.0569583, -0.1554351, 0.1555482

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554022, upper bound: 0.0551756
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554022, upper bound: 0.0551756
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0193350, 0.0216215, -0.0385351, 0.0371592
1: -0.0172633, 0.0331967, -0.0221682, 0.0413110, -0.0585744, 0.0553650
2: -0.0456765, 0.0224354, -0.0484735, 0.0279975, -0.0736740, 0.0709089
3: -0.0304100, 0.0416375, -0.0360067, 0.0530757, -0.0834857, 0.0776442
4: -0.0582723, 0.0274961, -0.0654610, 0.0313325, -0.0896048, 0.0929571

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 7

Time for candidate selection: 2.76 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535715, upper bound: 0.0546597
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549466, upper bound: 0.0551639
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0247696, 0.0287602, -0.0462448, 0.0432528
1: -0.0177545, 0.0346891, -0.0299947, 0.0641933, -0.0819479, 0.0646838
2: -0.0469266, 0.0241803, -0.0636112, 0.0451371, -0.0920638, 0.0877915
3: -0.0308810, 0.0437489, -0.0427042, 0.0836345, -0.1145155, 0.0864531
4: -0.0604094, 0.0290925, -0.0908074, 0.0485188, -0.1089282, 0.1198999

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551154, upper bound: 0.0552620
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551154, upper bound: 0.0554010
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0249527, 0.0289441, -0.0458578, 0.0427769
1: -0.0172633, 0.0331967, -0.0302124, 0.0647534, -0.0820168, 0.0634091
2: -0.0456765, 0.0224354, -0.0640738, 0.0455330, -0.0912095, 0.0865092
3: -0.0304100, 0.0416375, -0.0429836, 0.0844141, -0.1148241, 0.0846211
4: -0.0582723, 0.0274961, -0.0914269, 0.0490248, -0.1072971, 0.1189231

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0552620
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0554029
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0193350, 0.0216215, -0.0489594, 0.0504656
1: -0.0321509, 0.0732026, -0.0221682, 0.0413110, -0.0734620, 0.0953708
2: -0.0714296, 0.0520608, -0.0484735, 0.0279975, -0.0994272, 0.1005343
3: -0.0465936, 0.0955957, -0.0360067, 0.0530757, -0.0996693, 0.1316024
4: -0.0984768, 0.0579753, -0.0654610, 0.0313325, -0.1298094, 0.1234363

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 7

Time for candidate selection: 2.79 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534120, upper bound: 0.0546353
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548279, upper bound: 0.0551453
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0247696, 0.0287602, -0.0567690, 0.0572170
1: -0.0325964, 0.0775581, -0.0299947, 0.0641933, -0.0967898, 0.1075529
2: -0.0725141, 0.0540359, -0.0636112, 0.0451371, -0.1176512, 0.1176471
3: -0.0464773, 0.1009725, -0.0427042, 0.0836345, -0.1301118, 0.1436767
4: -0.1008744, 0.0594039, -0.0908074, 0.0485188, -0.1493932, 0.1502113

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551056, upper bound: 0.0550615
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551056, upper bound: 0.0553823
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0249527, 0.0289441, -0.0562820, 0.0560833
1: -0.0321509, 0.0732026, -0.0302124, 0.0647534, -0.0969044, 0.1034149
2: -0.0714296, 0.0520608, -0.0640738, 0.0455330, -0.1169626, 0.1161346
3: -0.0465936, 0.0955957, -0.0429836, 0.0844141, -0.1310077, 0.1385793
4: -0.0984768, 0.0579753, -0.0914269, 0.0490248, -0.1475016, 0.1494022

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551868, upper bound: 0.0550615
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551868, upper bound: 0.0553823
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0173510, 0.0182791, -0.0427603, 0.0457471
1: -0.0294868, 0.0633977, -0.0178209, 0.0343542, -0.0638410, 0.0812185
2: -0.0631345, 0.0446851, -0.0466004, 0.0233044, -0.0864388, 0.0912855
3: -0.0421317, 0.0824423, -0.0310903, 0.0432695, -0.0854012, 0.1135326
4: -0.0901240, 0.0480672, -0.0593948, 0.0285069, -0.1186308, 0.1074620

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554010, upper bound: 0.0551154
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554010, upper bound: 0.0553428
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0193350, 0.0216215, -0.0461026, 0.0477310
1: -0.0294868, 0.0633977, -0.0221682, 0.0413110, -0.0707978, 0.0855659
2: -0.0631345, 0.0446851, -0.0484735, 0.0279975, -0.0911320, 0.0931586
3: -0.0421317, 0.0824423, -0.0360067, 0.0530757, -0.0952074, 0.1184490
4: -0.0901240, 0.0480672, -0.0654610, 0.0313325, -0.1214565, 0.1135281

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 7

Time for candidate selection: 2.81 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536329, upper bound: 0.0546293
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550091, upper bound: 0.0551043
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0411523, 0.0514543, -0.0772170, 0.0716988
1: -0.0317584, 0.0698348, -0.0622191, 0.1311877, -0.1629460, 0.1320539
2: -0.0655043, 0.0478268, -0.0964486, 0.0705298, -0.1360340, 0.1442754
3: -0.0441914, 0.0912354, -0.0882282, 0.1948885, -0.2390799, 0.1794637
4: -0.0945988, 0.0507568, -0.1589930, 0.0784420, -0.1730408, 0.2097498

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552643, upper bound: 0.0551154
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552643, upper bound: 0.0553428
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0414462, 0.0518560, -0.0763372, 0.0698423
1: -0.0294868, 0.0633977, -0.0627760, 0.1324598, -0.1619466, 0.1261737
2: -0.0631345, 0.0446851, -0.0970791, 0.0711206, -0.1342551, 0.1417642
3: -0.0421317, 0.0824423, -0.0889659, 0.1968575, -0.2389892, 0.1714082
4: -0.0901240, 0.0480672, -0.1602575, 0.0791221, -0.1692460, 0.2083246

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554034, upper bound: 0.0551154
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554034, upper bound: 0.0553428
time: 0.32 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0235205, 0.0274677, -0.0532303, 0.0540670
1: -0.0317584, 0.0698348, -0.0268073, 0.0602619, -0.0920203, 0.0966421
2: -0.0655043, 0.0478268, -0.0614571, 0.0435361, -0.1090404, 0.1092838
3: -0.0441914, 0.0912354, -0.0389977, 0.0776374, -0.1218288, 0.1302331
4: -0.0945988, 0.0507568, -0.0874355, 0.0469115, -0.1415103, 0.1381923

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552474, upper bound: 0.0550916
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552474, upper bound: 0.0553391
time: 0.32 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0237000, 0.0276457, -0.0521268, 0.0520961
1: -0.0294868, 0.0633977, -0.0270187, 0.0608168, -0.0903036, 0.0904164
2: -0.0631345, 0.0446851, -0.0619147, 0.0439293, -0.1070638, 0.1065998
3: -0.0421317, 0.0824423, -0.0392698, 0.0784101, -0.1205418, 0.1217121
4: -0.0901240, 0.0480672, -0.0880449, 0.0474173, -0.1375412, 0.1361120

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553937, upper bound: 0.0550916
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553937, upper bound: 0.0553391
time: 0.36 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.83 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0552057, upper bound: 0.0552057
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0552057, upper bound: 0.0554091
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0554091, upper bound: 0.0552057
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0554091, upper bound: 0.0554129
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0551756, upper bound: 0.0552057
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0551756, upper bound: 0.0554091
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0554022, upper bound: 0.0552057
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0554022, upper bound: 0.0554126
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0552057, upper bound: 0.0551756
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0552057, upper bound: 0.0554022
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0554091, upper bound: 0.0551756
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0554091, upper bound: 0.0551756
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0551756, upper bound: 0.0551756
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0551756, upper bound: 0.0554022
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0554022, upper bound: 0.0551756
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0554022, upper bound: 0.0551756
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0535715, upper bound: 0.0546597
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0549466, upper bound: 0.0551639
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0551154, upper bound: 0.0552620
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0551154, upper bound: 0.0554010
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0552620
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0554029
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0534120, upper bound: 0.0546353
IS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0548279, upper bound: 0.0551453
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0551056, upper bound: 0.0550615
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0551056, upper bound: 0.0553823
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0551868, upper bound: 0.0550615
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0551868, upper bound: 0.0553823
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0554010, upper bound: 0.0551154
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0554010, upper bound: 0.0553428
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0536329, upper bound: 0.0546293
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0550091, upper bound: 0.0551043
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0552643, upper bound: 0.0551154
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0552643, upper bound: 0.0553428
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0554034, upper bound: 0.0551154
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0554034, upper bound: 0.0553428
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0552474, upper bound: 0.0550916
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0552474, upper bound: 0.0553391
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0553937, upper bound: 0.0550916
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -0.0553937, upper bound: 0.0553391

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0174846, 0.0184832, -0.0359678, 0.0359678
1: -0.0177545, 0.0346891, -0.0177545, 0.0346891, -0.0524436, 0.0524436
2: -0.0469266, 0.0241803, -0.0469266, 0.0241803, -0.0711069, 0.0711069
3: -0.0308810, 0.0437489, -0.0308810, 0.0437489, -0.0746299, 0.0746299
4: -0.0604094, 0.0290925, -0.0604094, 0.0290925, -0.0895019, 0.0895019

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0552692
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544429
time: 0.30 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0168805, 0.0177837, -0.0352684, 0.0353637
1: -0.0177545, 0.0346891, -0.0172204, 0.0330732, -0.0508277, 0.0519095
2: -0.0469266, 0.0241803, -0.0455946, 0.0223437, -0.0692703, 0.0697750
3: -0.0308810, 0.0437489, -0.0303572, 0.0414705, -0.0723515, 0.0741061
4: -0.0604094, 0.0290925, -0.0581362, 0.0274024, -0.0878118, 0.0872287

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0554006
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0547986
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0174846, 0.0184832, -0.0353969, 0.0353089
1: -0.0172633, 0.0331967, -0.0177545, 0.0346891, -0.0519524, 0.0509512
2: -0.0456765, 0.0224354, -0.0469266, 0.0241803, -0.0698568, 0.0693620
3: -0.0304100, 0.0416375, -0.0308810, 0.0437489, -0.0741589, 0.0725185
4: -0.0582723, 0.0274961, -0.0604094, 0.0290925, -0.0873647, 0.0879055

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536038, upper bound: 0.0527974
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553941, upper bound: 0.0551896
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0169136, 0.0178243, -0.0347379, 0.0347379
1: -0.0172633, 0.0331967, -0.0172633, 0.0331967, -0.0504600, 0.0504600
2: -0.0456765, 0.0224354, -0.0456765, 0.0224354, -0.0681119, 0.0681119
3: -0.0304100, 0.0416375, -0.0304100, 0.0416375, -0.0720475, 0.0720475
4: -0.0582723, 0.0274961, -0.0582723, 0.0274961, -0.0857684, 0.0857684

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536038, upper bound: 0.0535517
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553941, upper bound: 0.0551896
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0278393, 0.0313050, -0.0487896, 0.0463225
1: -0.0177545, 0.0346891, -0.0320025, 0.0745231, -0.0922776, 0.0666916
2: -0.0469266, 0.0241803, -0.0715446, 0.0517503, -0.0986769, 0.0957249
3: -0.0308810, 0.0437489, -0.0457141, 0.0967271, -0.1276081, 0.0894630
4: -0.0604094, 0.0290925, -0.0984029, 0.0572466, -0.1176560, 0.1274954

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548878, upper bound: 0.0553089
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0544692
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0272046, 0.0301253, -0.0476099, 0.0456878
1: -0.0177545, 0.0346891, -0.0316691, 0.0706761, -0.0884306, 0.0663582
2: -0.0469266, 0.0241803, -0.0705942, 0.0499720, -0.0968986, 0.0947745
3: -0.0308810, 0.0437489, -0.0459514, 0.0921504, -0.1230314, 0.0897003
4: -0.0604094, 0.0290925, -0.0963448, 0.0559998, -0.1164092, 0.1254373

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548878, upper bound: 0.0554006
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0547986
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0278393, 0.0313050, -0.0482186, 0.0456636
1: -0.0172633, 0.0331967, -0.0320025, 0.0745231, -0.0917864, 0.0651993
2: -0.0456765, 0.0224354, -0.0715446, 0.0517503, -0.0974268, 0.0939800
3: -0.0304100, 0.0416375, -0.0457141, 0.0967271, -0.1271372, 0.0873517
4: -0.0582723, 0.0274961, -0.0984029, 0.0572466, -0.1155189, 0.1258991

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535978, upper bound: 0.0527974
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553809, upper bound: 0.0551896
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0272046, 0.0301253, -0.0470389, 0.0450289
1: -0.0172633, 0.0331967, -0.0316691, 0.0706761, -0.0879394, 0.0648659
2: -0.0456765, 0.0224354, -0.0705942, 0.0499720, -0.0956485, 0.0930296
3: -0.0304100, 0.0416375, -0.0459514, 0.0921504, -0.1225604, 0.0875890
4: -0.0582723, 0.0274961, -0.0963448, 0.0559998, -0.1142721, 0.1238410

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535978, upper bound: 0.0535107
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553809, upper bound: 0.0552287
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0174846, 0.0184832, -0.0464920, 0.0499320
1: -0.0325964, 0.0775581, -0.0177545, 0.0346891, -0.0672855, 0.0953127
2: -0.0725141, 0.0540359, -0.0469266, 0.0241803, -0.0966944, 0.1009625
3: -0.0464773, 0.1009725, -0.0308810, 0.0437489, -0.0902262, 0.1318535
4: -0.1008744, 0.0594039, -0.0604094, 0.0290925, -0.1299669, 0.1198133

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546251, upper bound: 0.0529548
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551896, upper bound: 0.0553559
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0168805, 0.0177837, -0.0457926, 0.0493279
1: -0.0325964, 0.0775581, -0.0172204, 0.0330732, -0.0656696, 0.0947786
2: -0.0725141, 0.0540359, -0.0455946, 0.0223437, -0.0948578, 0.0996305
3: -0.0464773, 0.1009725, -0.0303572, 0.0414705, -0.0879478, 0.1313297
4: -0.1008744, 0.0594039, -0.0581362, 0.0274024, -0.1282768, 0.1175401

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546251, upper bound: 0.0533919
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551896, upper bound: 0.0553871
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0174846, 0.0184832, -0.0458211, 0.0486153
1: -0.0321509, 0.0732026, -0.0177545, 0.0346891, -0.0668400, 0.0909571
2: -0.0714296, 0.0520608, -0.0469266, 0.0241803, -0.0956100, 0.0989874
3: -0.0465936, 0.0955957, -0.0308810, 0.0437489, -0.0903425, 0.1264767
4: -0.0984768, 0.0579753, -0.0604094, 0.0290925, -0.1275693, 0.1183847

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545810, upper bound: 0.0529548
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553941, upper bound: 0.0551469
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0168805, 0.0177837, -0.0451216, 0.0480111
1: -0.0321509, 0.0732026, -0.0172204, 0.0330732, -0.0652241, 0.0904230
2: -0.0714296, 0.0520608, -0.0455946, 0.0223437, -0.0937734, 0.0976554
3: -0.0465936, 0.0955957, -0.0303572, 0.0414705, -0.0880641, 0.1259529
4: -0.0984768, 0.0579753, -0.0581362, 0.0274024, -0.1258792, 0.1161115

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545810, upper bound: 0.0535649
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553941, upper bound: 0.0551469
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0278393, 0.0313050, -0.0593138, 0.0602867
1: -0.0325964, 0.0775581, -0.0320025, 0.0745231, -0.1071195, 0.1095607
2: -0.0725141, 0.0540359, -0.0715446, 0.0517503, -0.1242644, 0.1255804
3: -0.0464773, 0.1009725, -0.0457141, 0.0967271, -0.1432044, 0.1466866
4: -0.1008744, 0.0594039, -0.0984029, 0.0572466, -0.1581210, 0.1578068

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548665, upper bound: 0.0529717
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551469, upper bound: 0.0553559
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0272046, 0.0301253, -0.0581341, 0.0596520
1: -0.0325964, 0.0775581, -0.0316691, 0.0706761, -0.1032725, 0.1092273
2: -0.0725141, 0.0540359, -0.0705942, 0.0499720, -0.1224861, 0.1246301
3: -0.0464773, 0.1009725, -0.0459514, 0.0921504, -0.1386277, 0.1469239
4: -0.1008744, 0.0594039, -0.0963448, 0.0559998, -0.1568742, 0.1557487

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548665, upper bound: 0.0533617
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551469, upper bound: 0.0553871
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0278393, 0.0313050, -0.0586428, 0.0589699
1: -0.0321509, 0.0732026, -0.0320025, 0.0745231, -0.1066740, 0.1052051
2: -0.0714296, 0.0520608, -0.0715446, 0.0517503, -0.1231800, 0.1236054
3: -0.0465936, 0.0955957, -0.0457141, 0.0967271, -0.1433207, 0.1413099
4: -0.0984768, 0.0579753, -0.0984029, 0.0572466, -0.1557235, 0.1563783

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548770, upper bound: 0.0529843
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553809, upper bound: 0.0551469
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0272046, 0.0301253, -0.0574632, 0.0583352
1: -0.0321509, 0.0732026, -0.0316691, 0.0706761, -0.1028270, 0.1048717
2: -0.0714296, 0.0520608, -0.0705942, 0.0499720, -0.1214016, 0.1226550
3: -0.0465936, 0.0955957, -0.0459514, 0.0921504, -0.1387440, 0.1415471
4: -0.0984768, 0.0579753, -0.0963448, 0.0559998, -0.1544766, 0.1543202

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548770, upper bound: 0.0535312
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553809, upper bound: 0.0551469
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0163063, 0.0177085, -0.0346221, 0.0341305
1: -0.0172633, 0.0331967, -0.0184658, 0.0330927, -0.0503560, 0.0516625
2: -0.0456765, 0.0224354, -0.0422703, 0.0221831, -0.0678596, 0.0647057
3: -0.0304100, 0.0416375, -0.0314417, 0.0417711, -0.0721811, 0.0730792
4: -0.0582723, 0.0274961, -0.0568412, 0.0248886, -0.0831609, 0.0843374

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0538431, upper bound: 0.0534840
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549118, upper bound: 0.0551402
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0257626, 0.0305466, -0.0480312, 0.0442459
1: -0.0177545, 0.0346891, -0.0317584, 0.0698348, -0.0875894, 0.0664474
2: -0.0469266, 0.0241803, -0.0655043, 0.0478268, -0.0947534, 0.0896846
3: -0.0308810, 0.0437489, -0.0441914, 0.0912354, -0.1221164, 0.0879403
4: -0.0604094, 0.0290925, -0.0945988, 0.0507568, -0.1111662, 0.1236912

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549722, upper bound: 0.0553124
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549439, upper bound: 0.0546063
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0244811, 0.0283961, -0.0458807, 0.0429644
1: -0.0177545, 0.0346891, -0.0294868, 0.0633977, -0.0811522, 0.0641759
2: -0.0469266, 0.0241803, -0.0631345, 0.0446851, -0.0916117, 0.0873148
3: -0.0308810, 0.0437489, -0.0421317, 0.0824423, -0.1133233, 0.0858805
4: -0.0604094, 0.0290925, -0.0901240, 0.0480672, -0.1084765, 0.1192164

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549722, upper bound: 0.0553124
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549439, upper bound: 0.0546063
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0257626, 0.0305466, -0.0474602, 0.0435869
1: -0.0172633, 0.0331967, -0.0317584, 0.0698348, -0.0870982, 0.0649551
2: -0.0456765, 0.0224354, -0.0655043, 0.0478268, -0.0935033, 0.0879397
3: -0.0304100, 0.0416375, -0.0441914, 0.0912354, -0.1216454, 0.0858289
4: -0.0582723, 0.0274961, -0.0945988, 0.0507568, -0.1090291, 0.1220949

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535021, upper bound: 0.0529719
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553184, upper bound: 0.0552430
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0244811, 0.0283961, -0.0453097, 0.0423054
1: -0.0172633, 0.0331967, -0.0294868, 0.0633977, -0.0806610, 0.0626835
2: -0.0456765, 0.0224354, -0.0631345, 0.0446851, -0.0903616, 0.0855699
3: -0.0304100, 0.0416375, -0.0421317, 0.0824423, -0.1128523, 0.0837692
4: -0.0582723, 0.0274961, -0.0901240, 0.0480672, -0.1063394, 0.1176201

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535021, upper bound: 0.0529719
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553184, upper bound: 0.0552430
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0244811, 0.0283961, -0.0564049, 0.0569285
1: -0.0325964, 0.0775581, -0.0294868, 0.0633977, -0.0959941, 0.1070449
2: -0.0725141, 0.0540359, -0.0631345, 0.0446851, -0.1171992, 0.1171703
3: -0.0464773, 0.1009725, -0.0421317, 0.0824423, -0.1289196, 0.1431042
4: -0.1008744, 0.0594039, -0.0901240, 0.0480672, -0.1489415, 0.1495278

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548612, upper bound: 0.0534798
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550756, upper bound: 0.0553594
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0257626, 0.0305466, -0.0578844, 0.0568933
1: -0.0321509, 0.0732026, -0.0317584, 0.0698348, -0.1019858, 0.1049610
2: -0.0714296, 0.0520608, -0.0655043, 0.0478268, -0.1192564, 0.1175651
3: -0.0465936, 0.0955957, -0.0441914, 0.0912354, -0.1378290, 0.1397871
4: -0.0984768, 0.0579753, -0.0945988, 0.0507568, -0.1492336, 0.1525741

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548770, upper bound: 0.0530085
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551382, upper bound: 0.0550371
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0244811, 0.0283961, -0.0557340, 0.0556118
1: -0.0321509, 0.0732026, -0.0294868, 0.0633977, -0.0955486, 0.1026894
2: -0.0714296, 0.0520608, -0.0631345, 0.0446851, -0.1161148, 0.1151953
3: -0.0465936, 0.0955957, -0.0421317, 0.0824423, -0.1290358, 0.1377274
4: -0.0984768, 0.0579753, -0.0901240, 0.0480672, -0.1465440, 0.1480993

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548770, upper bound: 0.0530085
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551382, upper bound: 0.0550371
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0174846, 0.0184832, -0.0429644, 0.0458807
1: -0.0294868, 0.0633977, -0.0177545, 0.0346891, -0.0641759, 0.0811522
2: -0.0631345, 0.0446851, -0.0469266, 0.0241803, -0.0873148, 0.0916117
3: -0.0421317, 0.0824423, -0.0308810, 0.0437489, -0.0858805, 0.1133233
4: -0.0901240, 0.0480672, -0.0604094, 0.0290925, -0.1192164, 0.1084765

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545488, upper bound: 0.0529833
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553821, upper bound: 0.0550854
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0168805, 0.0177837, -0.0422649, 0.0452766
1: -0.0294868, 0.0633977, -0.0172204, 0.0330732, -0.0625599, 0.0806181
2: -0.0631345, 0.0446851, -0.0455946, 0.0223437, -0.0854782, 0.0902798
3: -0.0421317, 0.0824423, -0.0303572, 0.0414705, -0.0836021, 0.1127995
4: -0.0901240, 0.0480672, -0.0581362, 0.0274024, -0.1175263, 0.1062034

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545488, upper bound: 0.0535778
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553821, upper bound: 0.0550854
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0424413, 0.0538162, -0.0795788, 0.0729878
1: -0.0317584, 0.0698348, -0.0657985, 0.1383410, -0.1700994, 0.1356333
2: -0.0655043, 0.0478268, -0.0985858, 0.0733478, -0.1388521, 0.1464126
3: -0.0441914, 0.0912354, -0.0931849, 0.2051075, -0.2492988, 0.1844203
4: -0.0945988, 0.0507568, -0.1642163, 0.0808722, -0.1754709, 0.2149731

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548794, upper bound: 0.0530411
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552431, upper bound: 0.0553015
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0409495, 0.0511984, -0.0769610, 0.0714961
1: -0.0317584, 0.0698348, -0.0619608, 0.1305476, -0.1623060, 0.1317956
2: -0.0655043, 0.0478268, -0.0960364, 0.0702451, -0.1357494, 0.1438631
3: -0.0441914, 0.0912354, -0.0879251, 0.1938453, -0.2380367, 0.1791605
4: -0.0945988, 0.0507568, -0.1583064, 0.0781185, -0.1727172, 0.2090632

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548794, upper bound: 0.0535772
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552431, upper bound: 0.0553193
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0424413, 0.0538162, -0.0782973, 0.0708373
1: -0.0294868, 0.0633977, -0.0657985, 0.1383410, -0.1678278, 0.1291962
2: -0.0631345, 0.0446851, -0.0985858, 0.0733478, -0.1364823, 0.1432709
3: -0.0421317, 0.0824423, -0.0931849, 0.2051075, -0.2472391, 0.1756271
4: -0.0901240, 0.0480672, -0.1642163, 0.0808722, -0.1709961, 0.2122834

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548727, upper bound: 0.0530411
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553821, upper bound: 0.0550854
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0409495, 0.0511984, -0.0756795, 0.0693456
1: -0.0294868, 0.0633977, -0.0619608, 0.1305476, -0.1600344, 0.1253585
2: -0.0631345, 0.0446851, -0.0960364, 0.0702451, -0.1333796, 0.1407215
3: -0.0421317, 0.0824423, -0.0879251, 0.1938453, -0.2359769, 0.1703673
4: -0.0901240, 0.0480672, -0.1583064, 0.0781185, -0.1682424, 0.2063736

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548727, upper bound: 0.0530411
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553821, upper bound: 0.0550854
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0240568, 0.0286650, -0.0544276, 0.0546034
1: -0.0317584, 0.0698348, -0.0273684, 0.0643119, -0.0960703, 0.0972032
2: -0.0655043, 0.0478268, -0.0624101, 0.0452202, -0.1107245, 0.1102369
3: -0.0441914, 0.0912354, -0.0390651, 0.0827163, -0.1269077, 0.1303005
4: -0.0945988, 0.0507568, -0.0896576, 0.0481224, -0.1427211, 0.1404144

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548794, upper bound: 0.0535113
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552222, upper bound: 0.0552889
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0232741, 0.0271414, -0.0529040, 0.0538207
1: -0.0317584, 0.0698348, -0.0263826, 0.0596271, -0.0913855, 0.0962174
2: -0.0655043, 0.0478268, -0.0610710, 0.0431272, -0.1086314, 0.1088977
3: -0.0441914, 0.0912354, -0.0385253, 0.0767525, -0.1209439, 0.1297607
4: -0.0945988, 0.0507568, -0.0868742, 0.0465096, -0.1411084, 0.1376310

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548794, upper bound: 0.0537691
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552222, upper bound: 0.0553146
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0240568, 0.0286650, -0.0531461, 0.0524529
1: -0.0294868, 0.0633977, -0.0273684, 0.0643119, -0.0937987, 0.0907660
2: -0.0631345, 0.0446851, -0.0624101, 0.0452202, -0.1083547, 0.1070952
3: -0.0421317, 0.0824423, -0.0390651, 0.0827163, -0.1248480, 0.1215073
4: -0.0901240, 0.0480672, -0.0896576, 0.0481224, -0.1382463, 0.1377247

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548764, upper bound: 0.0534960
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553693, upper bound: 0.0550641
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0232741, 0.0271414, -0.0516225, 0.0516702
1: -0.0294868, 0.0633977, -0.0263826, 0.0596271, -0.0891139, 0.0897803
2: -0.0631345, 0.0446851, -0.0610710, 0.0431272, -0.1062616, 0.1057561
3: -0.0421317, 0.0824423, -0.0385253, 0.0767525, -0.1188841, 0.1209676
4: -0.0901240, 0.0480672, -0.0868742, 0.0465096, -0.1366336, 0.1349413

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548764, upper bound: 0.0536437
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553693, upper bound: 0.0550641
time: 0.37 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.18 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0552692
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544429
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0554006
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0547986
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0536038, upper bound: 0.0527974
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0553941, upper bound: 0.0551896
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0536038, upper bound: 0.0535517
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0553941, upper bound: 0.0551896
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548878, upper bound: 0.0553089
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0544692
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548878, upper bound: 0.0554006
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0547986
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0535978, upper bound: 0.0527974
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0553809, upper bound: 0.0551896
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0535978, upper bound: 0.0535107
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0553809, upper bound: 0.0552287
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0546251, upper bound: 0.0529548
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0551896, upper bound: 0.0553559
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0546251, upper bound: 0.0533919
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0551896, upper bound: 0.0553871
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0545810, upper bound: 0.0529548
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0553941, upper bound: 0.0551469
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0545810, upper bound: 0.0535649
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0553941, upper bound: 0.0551469
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548665, upper bound: 0.0529717
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0551469, upper bound: 0.0553559
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548665, upper bound: 0.0533617
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0551469, upper bound: 0.0553871
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548770, upper bound: 0.0529843
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0553809, upper bound: 0.0551469
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548770, upper bound: 0.0535312
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0553809, upper bound: 0.0551469
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0538431, upper bound: 0.0534840
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0549118, upper bound: 0.0551402
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0549722, upper bound: 0.0553124
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0549439, upper bound: 0.0546063
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0549722, upper bound: 0.0553124
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0549439, upper bound: 0.0546063
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0535021, upper bound: 0.0529719
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0553184, upper bound: 0.0552430
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0535021, upper bound: 0.0529719
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0553184, upper bound: 0.0552430
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548612, upper bound: 0.0534798
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0550756, upper bound: 0.0553594
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548770, upper bound: 0.0530085
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0551382, upper bound: 0.0550371
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548770, upper bound: 0.0530085
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0551382, upper bound: 0.0550371
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0545488, upper bound: 0.0529833
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0553821, upper bound: 0.0550854
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0545488, upper bound: 0.0535778
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0553821, upper bound: 0.0550854
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548794, upper bound: 0.0530411
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0552431, upper bound: 0.0553015
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548794, upper bound: 0.0535772
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0552431, upper bound: 0.0553193
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548727, upper bound: 0.0530411
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0553821, upper bound: 0.0550854
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548727, upper bound: 0.0530411
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0553821, upper bound: 0.0550854
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548794, upper bound: 0.0535113
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0552222, upper bound: 0.0552889
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548794, upper bound: 0.0537691
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0552222, upper bound: 0.0553146
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548764, upper bound: 0.0534960
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0553693, upper bound: 0.0550641
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0548764, upper bound: 0.0536437
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.18
Output dim: 0, lower bound: -0.0553693, upper bound: 0.0550641

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0173068, 0.0182680, -0.0371008, 0.0364247
1: -0.0185345, 0.0336902, -0.0173607, 0.0339652, -0.0524997, 0.0510508
2: -0.0465600, 0.0245035, -0.0465282, 0.0238339, -0.0703939, 0.0710317
3: -0.0314490, 0.0414548, -0.0303206, 0.0427798, -0.0742288, 0.0717754
4: -0.0564701, 0.0289175, -0.0598727, 0.0286871, -0.0851573, 0.0887902

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544429
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544429
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0166915, 0.0175625, -0.0363953, 0.0358093
1: -0.0185345, 0.0336902, -0.0168020, 0.0323275, -0.0508620, 0.0504922
2: -0.0465600, 0.0245035, -0.0451872, 0.0219984, -0.0685584, 0.0696907
3: -0.0314490, 0.0414548, -0.0297637, 0.0404095, -0.0718586, 0.0712185
4: -0.0564701, 0.0289175, -0.0575868, 0.0269948, -0.0834650, 0.0865043

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0527675, upper bound: 0.0534955
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549181, upper bound: 0.0553872
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0163001, 0.0171033, -0.0174846, 0.0184832, -0.0347834, 0.0345879
1: -0.0163531, 0.0312095, -0.0177545, 0.0346891, -0.0510422, 0.0489641
2: -0.0443946, 0.0211404, -0.0469266, 0.0241803, -0.0685749, 0.0680670
3: -0.0292436, 0.0388854, -0.0308810, 0.0437489, -0.0729924, 0.0697664
4: -0.0566717, 0.0260474, -0.0604094, 0.0290925, -0.0857642, 0.0864568

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553872, upper bound: 0.0549181
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546909, upper bound: 0.0548758
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0163001, 0.0171033, -0.0169136, 0.0178243, -0.0341244, 0.0340169
1: -0.0163531, 0.0312095, -0.0172633, 0.0331967, -0.0495499, 0.0484729
2: -0.0443946, 0.0211404, -0.0456765, 0.0224354, -0.0668300, 0.0668169
3: -0.0292436, 0.0388854, -0.0304100, 0.0416375, -0.0708811, 0.0692954
4: -0.0566717, 0.0260474, -0.0582723, 0.0274961, -0.0841679, 0.0843197

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536317, upper bound: 0.0548962
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0536317, upper bound: 0.0552287
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0277450, 0.0312228, -0.0500557, 0.0468628
1: -0.0185345, 0.0336902, -0.0317503, 0.0742458, -0.0927803, 0.0654405
2: -0.0465600, 0.0245035, -0.0713875, 0.0516147, -0.0981747, 0.0958910
3: -0.0314490, 0.0414548, -0.0454424, 0.0962967, -0.1277457, 0.0868972
4: -0.0564701, 0.0289175, -0.0981912, 0.0570942, -0.1135643, 0.1271087

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0544692
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0544692
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0270998, 0.0300402, -0.0488730, 0.0462176
1: -0.0185345, 0.0336902, -0.0313774, 0.0702919, -0.0888264, 0.0650675
2: -0.0465600, 0.0245035, -0.0704184, 0.0498419, -0.0964019, 0.0949220
3: -0.0314490, 0.0414548, -0.0456503, 0.0915263, -0.1229754, 0.0871051
4: -0.0564701, 0.0289175, -0.0960980, 0.0558524, -0.1123225, 0.1250155

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529254, upper bound: 0.0545499
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549521, upper bound: 0.0553872
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0163001, 0.0171033, -0.0278393, 0.0313050, -0.0476051, 0.0449426
1: -0.0163531, 0.0312095, -0.0320025, 0.0745231, -0.0908762, 0.0632121
2: -0.0443946, 0.0211404, -0.0715446, 0.0517503, -0.0961449, 0.0926850
3: -0.0292436, 0.0388854, -0.0457141, 0.0967271, -0.1259707, 0.0845995
4: -0.0566717, 0.0260474, -0.0984029, 0.0572466, -0.1139184, 0.1244503

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533919, upper bound: 0.0548407
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0533919, upper bound: 0.0551896
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0163001, 0.0171033, -0.0272046, 0.0301253, -0.0464254, 0.0443079
1: -0.0163531, 0.0312095, -0.0316691, 0.0706761, -0.0870292, 0.0628787
2: -0.0443946, 0.0211404, -0.0705942, 0.0499720, -0.0943666, 0.0917346
3: -0.0292436, 0.0388854, -0.0459514, 0.0921504, -0.1213940, 0.0848368
4: -0.0566717, 0.0260474, -0.0963448, 0.0559998, -0.1126716, 0.1223922

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536523, upper bound: 0.0549016
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0536523, upper bound: 0.0552287
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0275401, 0.0318847, -0.0174846, 0.0184832, -0.0460233, 0.0493693
1: -0.0315660, 0.0759994, -0.0177545, 0.0346891, -0.0662550, 0.0937540
2: -0.0714210, 0.0530449, -0.0469266, 0.0241803, -0.0956013, 0.0999715
3: -0.0452617, 0.0986971, -0.0308810, 0.0437489, -0.0890106, 0.1295781
4: -0.0993945, 0.0582935, -0.0604094, 0.0290925, -0.1284870, 0.1187029

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552856, upper bound: 0.0548459
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543515, upper bound: 0.0548036
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0275401, 0.0318847, -0.0168805, 0.0177837, -0.0453238, 0.0487651
1: -0.0315660, 0.0759994, -0.0172204, 0.0330732, -0.0646391, 0.0932199
2: -0.0714210, 0.0530449, -0.0455946, 0.0223437, -0.0937647, 0.0986395
3: -0.0452617, 0.0986971, -0.0303572, 0.0414705, -0.0867322, 0.1290543
4: -0.0993945, 0.0582935, -0.0581362, 0.0274024, -0.1267969, 0.1164297

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0527974, upper bound: 0.0535978
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0527974, upper bound: 0.0553871
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0268415, 0.0305504, -0.0174846, 0.0184832, -0.0453247, 0.0480350
1: -0.0310880, 0.0714282, -0.0177545, 0.0346891, -0.0657771, 0.0891827
2: -0.0703224, 0.0510674, -0.0469266, 0.0241803, -0.0945028, 0.0979940
3: -0.0453250, 0.0927791, -0.0308810, 0.0437489, -0.0890739, 0.1236601
4: -0.0969780, 0.0568669, -0.0604094, 0.0290925, -0.1260705, 0.1172763

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553872, upper bound: 0.0549521
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546909, upper bound: 0.0549097
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0268415, 0.0305504, -0.0168805, 0.0177837, -0.0446252, 0.0474309
1: -0.0310880, 0.0714282, -0.0172204, 0.0330732, -0.0641612, 0.0886486
2: -0.0703224, 0.0510674, -0.0455946, 0.0223437, -0.0926662, 0.0966620
3: -0.0453250, 0.0927791, -0.0303572, 0.0414705, -0.0867955, 0.1231363
4: -0.0969780, 0.0568669, -0.0581362, 0.0274024, -0.1243804, 0.1150032

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534829, upper bound: 0.0548581
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534829, upper bound: 0.0551469
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0275401, 0.0318847, -0.0278393, 0.0313050, -0.0588451, 0.0597239
1: -0.0315660, 0.0759994, -0.0320025, 0.0745231, -0.1060890, 0.1080020
2: -0.0714210, 0.0530449, -0.0715446, 0.0517503, -0.1231713, 0.1245894
3: -0.0452617, 0.0986971, -0.0457141, 0.0967271, -0.1419889, 0.1444112
4: -0.0993945, 0.0582935, -0.0984029, 0.0572466, -0.1566412, 0.1566964

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0528883, upper bound: 0.0544393
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0528883, upper bound: 0.0553559
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0275401, 0.0318847, -0.0272046, 0.0301253, -0.0576654, 0.0590893
1: -0.0315660, 0.0759994, -0.0316691, 0.0706761, -0.1022421, 0.1076686
2: -0.0714210, 0.0530449, -0.0705942, 0.0499720, -0.1213930, 0.1236390
3: -0.0452617, 0.0986971, -0.0459514, 0.0921504, -0.1374122, 0.1446485
4: -0.0993945, 0.0582935, -0.0963448, 0.0559998, -0.1553943, 0.1546383

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529646, upper bound: 0.0547068
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529646, upper bound: 0.0547068
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0268415, 0.0305504, -0.0278393, 0.0313050, -0.0581465, 0.0583897
1: -0.0310880, 0.0714282, -0.0320025, 0.0745231, -0.1056111, 0.1034307
2: -0.0703224, 0.0510674, -0.0715446, 0.0517503, -0.1220728, 0.1226119
3: -0.0453250, 0.0927791, -0.0457141, 0.0967271, -0.1420521, 0.1384932
4: -0.0969780, 0.0568669, -0.0984029, 0.0572466, -0.1542247, 0.1552699

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529324, upper bound: 0.0539344
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529324, upper bound: 0.0551469
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0268415, 0.0305504, -0.0272046, 0.0301253, -0.0569668, 0.0577550
1: -0.0310880, 0.0714282, -0.0316691, 0.0706761, -0.1017641, 0.1030973
2: -0.0703224, 0.0510674, -0.0705942, 0.0499720, -0.1202944, 0.1216615
3: -0.0453250, 0.0927791, -0.0459514, 0.0921504, -0.1374754, 0.1387305
4: -0.0969780, 0.0568669, -0.0963448, 0.0559998, -0.1529779, 0.1532118

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534829, upper bound: 0.0548593
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534829, upper bound: 0.0548593
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0256414, 0.0304423, -0.0492751, 0.0447593
1: -0.0185345, 0.0336902, -0.0314712, 0.0694659, -0.0880005, 0.0651614
2: -0.0465600, 0.0245035, -0.0653138, 0.0476777, -0.0942377, 0.0898173
3: -0.0314490, 0.0414548, -0.0438639, 0.0906439, -0.1220929, 0.0853187
4: -0.0564701, 0.0289175, -0.0943035, 0.0505933, -0.1070634, 0.1232210

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529517, upper bound: 0.0547784
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549449, upper bound: 0.0552997
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0243660, 0.0282986, -0.0471315, 0.0434838
1: -0.0185345, 0.0336902, -0.0292173, 0.0630555, -0.0815900, 0.0629075
2: -0.0465600, 0.0245035, -0.0629609, 0.0445484, -0.0911084, 0.0874644
3: -0.0314490, 0.0414548, -0.0418369, 0.0818994, -0.1133485, 0.0832917
4: -0.0564701, 0.0289175, -0.0898552, 0.0479190, -0.1043891, 0.1187727

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529517, upper bound: 0.0545173
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549521, upper bound: 0.0553522
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0163001, 0.0171033, -0.0257626, 0.0305466, -0.0468467, 0.0428659
1: -0.0163531, 0.0312095, -0.0317584, 0.0698348, -0.0861880, 0.0629679
2: -0.0443946, 0.0211404, -0.0655043, 0.0478268, -0.0922213, 0.0866447
3: -0.0292436, 0.0388854, -0.0441914, 0.0912354, -0.1204790, 0.0830768
4: -0.0566717, 0.0260474, -0.0945988, 0.0507568, -0.1074286, 0.1206462

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535791, upper bound: 0.0548549
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0535791, upper bound: 0.0552430
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0163001, 0.0171033, -0.0244811, 0.0283961, -0.0446962, 0.0415844
1: -0.0163531, 0.0312095, -0.0294868, 0.0633977, -0.0797508, 0.0606963
2: -0.0443946, 0.0211404, -0.0631345, 0.0446851, -0.0890797, 0.0842749
3: -0.0292436, 0.0388854, -0.0421317, 0.0824423, -0.1116858, 0.0810170
4: -0.0566717, 0.0260474, -0.0901240, 0.0480672, -0.1047389, 0.1161713

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546124, upper bound: 0.0551136
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553184, upper bound: 0.0552430
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0275401, 0.0318847, -0.0244811, 0.0283961, -0.0559362, 0.0563658
1: -0.0315660, 0.0759994, -0.0294868, 0.0633977, -0.0949637, 0.1054862
2: -0.0714210, 0.0530449, -0.0631345, 0.0446851, -0.1161061, 0.1161793
3: -0.0452617, 0.0986971, -0.0421317, 0.0824423, -0.1277040, 0.1408287
4: -0.0993945, 0.0582935, -0.0901240, 0.0480672, -0.1474617, 0.1484174

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529686, upper bound: 0.0543953
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529686, upper bound: 0.0543953
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0238435, 0.0277011, -0.0174846, 0.0184832, -0.0423268, 0.0451857
1: -0.0284588, 0.0615482, -0.0177545, 0.0346891, -0.0631479, 0.0793027
2: -0.0619252, 0.0435389, -0.0469266, 0.0241803, -0.0861056, 0.0904655
3: -0.0407595, 0.0795807, -0.0308810, 0.0437489, -0.0845083, 0.1104617
4: -0.0884067, 0.0467996, -0.0604094, 0.0290925, -0.1174992, 0.1072090

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553522, upper bound: 0.0549521
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547055, upper bound: 0.0549097
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0238435, 0.0277011, -0.0168805, 0.0177837, -0.0416273, 0.0445815
1: -0.0284588, 0.0615482, -0.0172204, 0.0330732, -0.0615320, 0.0787686
2: -0.0619252, 0.0435389, -0.0455946, 0.0223437, -0.0842690, 0.0891336
3: -0.0407595, 0.0795807, -0.0303572, 0.0414705, -0.0822299, 0.1099379
4: -0.0884067, 0.0467996, -0.0581362, 0.0274024, -0.1158091, 0.1049358

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536887, upper bound: 0.0548305
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536887, upper bound: 0.0550854
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0251495, 0.0299018, -0.0424413, 0.0538162, -0.0789656, 0.0723431
1: -0.0308103, 0.0680934, -0.0657985, 0.1383410, -0.1691514, 0.1338919
2: -0.0643693, 0.0467766, -0.0985858, 0.0733478, -0.1377171, 0.1453625
3: -0.0430202, 0.0885118, -0.0931849, 0.2051075, -0.2481277, 0.1816966
4: -0.0929886, 0.0495827, -0.1642163, 0.0808722, -0.1738608, 0.2137989

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0532313, upper bound: 0.0548078
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0532313, upper bound: 0.0553015
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0251495, 0.0299018, -0.0409495, 0.0511984, -0.0763478, 0.0708514
1: -0.0308103, 0.0680934, -0.0619608, 0.1305476, -0.1613580, 0.1300542
2: -0.0643693, 0.0467766, -0.0960364, 0.0702451, -0.1346144, 0.1428130
3: -0.0430202, 0.0885118, -0.0879251, 0.1938453, -0.2368655, 0.1764368
4: -0.0929886, 0.0495827, -0.1583064, 0.0781185, -0.1711071, 0.2078891

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0532437, upper bound: 0.0548167
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0532437, upper bound: 0.0548167
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0238435, 0.0277011, -0.0424413, 0.0538162, -0.0776597, 0.0701423
1: -0.0284588, 0.0615482, -0.0657985, 0.1383410, -0.1667999, 0.1273467
2: -0.0619252, 0.0435389, -0.0985858, 0.0733478, -0.1352731, 0.1421247
3: -0.0407595, 0.0795807, -0.0931849, 0.2051075, -0.2458669, 0.1727656
4: -0.0884067, 0.0467996, -0.1642163, 0.0808722, -0.1692789, 0.2110158

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534883, upper bound: 0.0547894
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534883, upper bound: 0.0547894
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0238435, 0.0277011, -0.0409495, 0.0511984, -0.0750419, 0.0686506
1: -0.0284588, 0.0615482, -0.0619608, 0.1305476, -0.1590064, 0.1235090
2: -0.0619252, 0.0435389, -0.0960364, 0.0702451, -0.1321704, 0.1395753
3: -0.0407595, 0.0795807, -0.0879251, 0.1938453, -0.2346047, 0.1675058
4: -0.0884067, 0.0467996, -0.1583064, 0.0781185, -0.1665252, 0.2051060

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537094, upper bound: 0.0548559
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537094, upper bound: 0.0550854
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0251495, 0.0299018, -0.0240568, 0.0286650, -0.0538144, 0.0539586
1: -0.0308103, 0.0680934, -0.0273684, 0.0643119, -0.0951223, 0.0954618
2: -0.0643693, 0.0467766, -0.0624101, 0.0452202, -0.1095895, 0.1091868
3: -0.0430202, 0.0885118, -0.0390651, 0.0827163, -0.1257365, 0.1275768
4: -0.0929886, 0.0495827, -0.0896576, 0.0481224, -0.1411110, 0.1392402

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535289, upper bound: 0.0548026
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535289, upper bound: 0.0548026
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0251495, 0.0299018, -0.0232741, 0.0271414, -0.0522909, 0.0531760
1: -0.0308103, 0.0680934, -0.0263826, 0.0596271, -0.0904374, 0.0944760
2: -0.0643693, 0.0467766, -0.0610710, 0.0431272, -0.1074964, 0.1078476
3: -0.0430202, 0.0885118, -0.0385253, 0.0767525, -0.1197727, 0.1270371
4: -0.0929886, 0.0495827, -0.0868742, 0.0465096, -0.1394983, 0.1364569

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535289, upper bound: 0.0548111
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0535289, upper bound: 0.0553146
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0238435, 0.0277011, -0.0240568, 0.0286650, -0.0525085, 0.0517579
1: -0.0284588, 0.0615482, -0.0273684, 0.0643119, -0.0927707, 0.0889166
2: -0.0619252, 0.0435389, -0.0624101, 0.0452202, -0.1071455, 0.1059490
3: -0.0407595, 0.0795807, -0.0390651, 0.0827163, -0.1234758, 0.1186457
4: -0.0884067, 0.0467996, -0.0896576, 0.0481224, -0.1365291, 0.1364571

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536540, upper bound: 0.0547787
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536540, upper bound: 0.0550641
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0238435, 0.0277011, -0.0232741, 0.0271414, -0.0509849, 0.0509752
1: -0.0284588, 0.0615482, -0.0263826, 0.0596271, -0.0880859, 0.0879308
2: -0.0619252, 0.0435389, -0.0610710, 0.0431272, -0.1050524, 0.1046099
3: -0.0407595, 0.0795807, -0.0385253, 0.0767525, -0.1175119, 0.1181060
4: -0.0884067, 0.0467996, -0.0868742, 0.0465096, -0.1349163, 0.1336738

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537749, upper bound: 0.0547965
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537749, upper bound: 0.0550641
time: 0.38 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.04 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544429
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544429
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0527675, upper bound: 0.0534955
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0549181, upper bound: 0.0553872
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0553872, upper bound: 0.0549181
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0546909, upper bound: 0.0548758
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0536317, upper bound: 0.0548962
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0536317, upper bound: 0.0552287
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0544692
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0544692
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0529254, upper bound: 0.0545499
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0549521, upper bound: 0.0553872
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0533919, upper bound: 0.0548407
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0533919, upper bound: 0.0551896
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0536523, upper bound: 0.0549016
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0536523, upper bound: 0.0552287
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0552856, upper bound: 0.0548459
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0543515, upper bound: 0.0548036
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0527974, upper bound: 0.0535978
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0527974, upper bound: 0.0553871
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0553872, upper bound: 0.0549521
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0546909, upper bound: 0.0549097
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0534829, upper bound: 0.0548581
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0534829, upper bound: 0.0551469
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0528883, upper bound: 0.0544393
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0528883, upper bound: 0.0553559
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0529646, upper bound: 0.0547068
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0529646, upper bound: 0.0547068
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0529324, upper bound: 0.0539344
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0529324, upper bound: 0.0551469
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0534829, upper bound: 0.0548593
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0534829, upper bound: 0.0548593
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0529517, upper bound: 0.0547784
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0549449, upper bound: 0.0552997
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0529517, upper bound: 0.0545173
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0549521, upper bound: 0.0553522
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0535791, upper bound: 0.0548549
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0535791, upper bound: 0.0552430
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0546124, upper bound: 0.0551136
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0553184, upper bound: 0.0552430
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0529686, upper bound: 0.0543953
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0529686, upper bound: 0.0543953
IS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0553522, upper bound: 0.0549521
IS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0547055, upper bound: 0.0549097
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0536887, upper bound: 0.0548305
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0536887, upper bound: 0.0550854
IS_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0532313, upper bound: 0.0548078
IS_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0532313, upper bound: 0.0553015
IS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0532437, upper bound: 0.0548167
IS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0532437, upper bound: 0.0548167
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0534883, upper bound: 0.0547894
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0534883, upper bound: 0.0547894
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0537094, upper bound: 0.0548559
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0537094, upper bound: 0.0550854
IS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0535289, upper bound: 0.0548026
IS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0535289, upper bound: 0.0548026
IS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0535289, upper bound: 0.0548111
IS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0535289, upper bound: 0.0553146
IS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0536540, upper bound: 0.0547787
IS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0536540, upper bound: 0.0550641
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0537749, upper bound: 0.0547965
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.04
Output dim: 0, lower bound: -0.0537749, upper bound: 0.0550641

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0160855, 0.0168588, -0.0356917, 0.0352034
1: -0.0185345, 0.0336902, -0.0158883, 0.0303832, -0.0489177, 0.0495785
2: -0.0465600, 0.0245035, -0.0439427, 0.0207548, -0.0673148, 0.0684462
3: -0.0314490, 0.0414548, -0.0285874, 0.0377832, -0.0692322, 0.0700422
4: -0.0564701, 0.0289175, -0.0560427, 0.0256004, -0.0820705, 0.0849601

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 7

Time for candidate selection: 2.10 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543623, upper bound: 0.0540256
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546596, upper bound: 0.0551596
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0161215, 0.0169014, -0.0188328, 0.0191179, -0.0352393, 0.0357342
1: -0.0159319, 0.0305109, -0.0185345, 0.0336902, -0.0496220, 0.0490454
2: -0.0440287, 0.0208520, -0.0465600, 0.0245035, -0.0685322, 0.0674120
3: -0.0286411, 0.0379530, -0.0314490, 0.0414548, -0.0700959, 0.0694020
4: -0.0561858, 0.0257001, -0.0564701, 0.0289175, -0.0851032, 0.0821703

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546242, upper bound: 0.0548758
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546242, upper bound: 0.0548758
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0163001, 0.0171033, -0.0163001, 0.0171033, -0.0334034, 0.0334034
1: -0.0163531, 0.0312095, -0.0163531, 0.0312095, -0.0475627, 0.0475627
2: -0.0443946, 0.0211404, -0.0443946, 0.0211404, -0.0655350, 0.0655350
3: -0.0292436, 0.0388854, -0.0292436, 0.0388854, -0.0681289, 0.0681289
4: -0.0566717, 0.0260474, -0.0566717, 0.0260474, -0.0827191, 0.0827191

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0533112, upper bound: 0.0552280
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536153, upper bound: 0.0551117
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0265997, 0.0294951, -0.0483279, 0.0457176
1: -0.0185345, 0.0336902, -0.0303341, 0.0685514, -0.0870859, 0.0640243
2: -0.0465600, 0.0245035, -0.0693358, 0.0488963, -0.0954563, 0.0938393
3: -0.0314490, 0.0414548, -0.0444010, 0.0887851, -0.1202342, 0.0858558
4: -0.0564701, 0.0289175, -0.0946760, 0.0547808, -0.1112510, 0.1235934

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 7

Time for candidate selection: 2.10 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543963, upper bound: 0.0540256
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546936, upper bound: 0.0551543
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0163001, 0.0171033, -0.0273750, 0.0307741, -0.0470743, 0.0444783
1: -0.0163531, 0.0312095, -0.0309930, 0.0730429, -0.0893961, 0.0622025
2: -0.0443946, 0.0211404, -0.0704783, 0.0508013, -0.0951959, 0.0916188
3: -0.0292436, 0.0388854, -0.0445269, 0.0944616, -0.1237051, 0.0834122
4: -0.0566717, 0.0260474, -0.0969963, 0.0561766, -0.1128483, 0.1230437

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0532562, upper bound: 0.0551784
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533306, upper bound: 0.0549130
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0163001, 0.0171033, -0.0267064, 0.0295812, -0.0458814, 0.0438097
1: -0.0163531, 0.0312095, -0.0306290, 0.0689026, -0.0852558, 0.0618385
2: -0.0443946, 0.0211404, -0.0695118, 0.0490279, -0.0934225, 0.0906522
3: -0.0292436, 0.0388854, -0.0447101, 0.0893868, -0.1186304, 0.0835955
4: -0.0566717, 0.0260474, -0.0949247, 0.0549288, -0.1116006, 0.1209721

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0533702, upper bound: 0.0552280
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536319, upper bound: 0.0548423
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0274449, 0.0318011, -0.0188328, 0.0191179, -0.0465627, 0.0506339
1: -0.0313060, 0.0757327, -0.0185345, 0.0336902, -0.0649961, 0.0942673
2: -0.0712640, 0.0529098, -0.0465600, 0.0245035, -0.0957675, 0.0994698
3: -0.0449828, 0.0982662, -0.0314490, 0.0414548, -0.0864376, 0.1297152
4: -0.0991794, 0.0581425, -0.0564701, 0.0289175, -0.1280969, 0.1146126

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543515, upper bound: 0.0548036
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543515, upper bound: 0.0548036
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0275401, 0.0318847, -0.0162643, 0.0170615, -0.0446015, 0.0481490
1: -0.0315660, 0.0759994, -0.0163098, 0.0310826, -0.0626486, 0.0923093
2: -0.0714210, 0.0530449, -0.0443089, 0.0210437, -0.0924647, 0.0973537
3: -0.0452617, 0.0986971, -0.0291901, 0.0387173, -0.0839790, 0.1278871
4: -0.0993945, 0.0582935, -0.0565295, 0.0259479, -0.1253425, 0.1148230

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 7

Time for candidate selection: 2.19 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0524936, upper bound: 0.0551145
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0526664, upper bound: 0.0535261
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0527075, upper bound: 0.0552739
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0267323, 0.0304591, -0.0188328, 0.0191179, -0.0458502, 0.0492920
1: -0.0307861, 0.0711128, -0.0185345, 0.0336902, -0.0644762, 0.0896473
2: -0.0701481, 0.0509288, -0.0465600, 0.0245035, -0.0946516, 0.0974888
3: -0.0450137, 0.0922674, -0.0314490, 0.0414548, -0.0864685, 0.1237164
4: -0.0967172, 0.0567164, -0.0564701, 0.0289175, -0.1256347, 0.1131865

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544728, upper bound: 0.0549097
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544728, upper bound: 0.0549097
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0275401, 0.0318847, -0.0273750, 0.0307741, -0.0583142, 0.0592597
1: -0.0315660, 0.0759994, -0.0309930, 0.0730429, -0.1046089, 0.1069924
2: -0.0714210, 0.0530449, -0.0704783, 0.0508013, -0.1222223, 0.1235232
3: -0.0452617, 0.0986971, -0.0445269, 0.0944616, -0.1397233, 0.1432239
4: -0.0993945, 0.0582935, -0.0969963, 0.0561766, -0.1555711, 0.1552898

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0528588, upper bound: 0.0552870
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0528586, upper bound: 0.0548381
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0250257, 0.0297954, -0.0486282, 0.0441436
1: -0.0185345, 0.0336902, -0.0305170, 0.0677239, -0.0862584, 0.0642071
2: -0.0465600, 0.0245035, -0.0641734, 0.0466243, -0.0931843, 0.0886769
3: -0.0314490, 0.0414548, -0.0426865, 0.0879095, -0.1193586, 0.0841413
4: -0.0564701, 0.0289175, -0.0926885, 0.0494157, -0.1058858, 0.1216060

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 7

Time for candidate selection: 2.16 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543891, upper bound: 0.0538933
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546864, upper bound: 0.0550182
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0237260, 0.0276022, -0.0464350, 0.0428439
1: -0.0185345, 0.0336902, -0.0281835, 0.0612032, -0.0797377, 0.0618737
2: -0.0465600, 0.0245035, -0.0617504, 0.0433997, -0.0899597, 0.0862539
3: -0.0314490, 0.0414548, -0.0404415, 0.0790431, -0.1104921, 0.0818963
4: -0.0564701, 0.0289175, -0.0881386, 0.0466485, -0.1031186, 0.1170561

Time for backsubstitution: 2.10 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0036636, high=0.0527477, mid=0.0527477, abs_max=0.058847926557064056
rel_dist={0: [-0.05567830804756887, 0.055678308047568875]}

## Binary search (step 2) starts
Candidate diff: 0.0282057


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0552486
time: 0.31 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552439, upper bound: 0.0552439
time: 0.28 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.75 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0552486
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -0.0552439, upper bound: 0.0552439

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0206600, 0.0216224, -0.0232866, 0.0246099, -0.0452699, 0.0449090
1: -0.0226827, 0.0442280, -0.0266501, 0.0523549, -0.0750376, 0.0708781
2: -0.0535189, 0.0294451, -0.0586758, 0.0341689, -0.0876878, 0.0881208
3: -0.0368305, 0.0571968, -0.0416921, 0.0688619, -0.1056924, 0.0988889
4: -0.0685483, 0.0351860, -0.0759283, 0.0403609, -0.1089091, 0.1111142

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552100, upper bound: 0.0552100
time: 0.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552100, upper bound: 0.0552439
time: 0.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0213640, 0.0238672, -0.0234919, 0.0252934, -0.0466574, 0.0473591
1: -0.0256137, 0.0480500, -0.0277824, 0.0525645, -0.0781783, 0.0758324
2: -0.0530172, 0.0317309, -0.0576905, 0.0337116, -0.0867288, 0.0894214
3: -0.0398649, 0.0629934, -0.0431061, 0.0695643, -0.1094292, 0.1060995
4: -0.0719933, 0.0355104, -0.0755393, 0.0389305, -0.1109238, 0.1110497

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551508, upper bound: 0.0551956
time: 0.30 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552439, upper bound: 0.0552439
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.56 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0552100, upper bound: 0.0552100
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0552100, upper bound: 0.0552439
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0551508, upper bound: 0.0551956
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0552439, upper bound: 0.0552439

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0206600, 0.0216224, -0.0206600, 0.0216224, -0.0422824, 0.0422824
1: -0.0226827, 0.0442280, -0.0226827, 0.0442280, -0.0669107, 0.0669107
2: -0.0535189, 0.0294451, -0.0535189, 0.0294451, -0.0829640, 0.0829640
3: -0.0368305, 0.0571968, -0.0368305, 0.0571968, -0.0940273, 0.0940273
4: -0.0685483, 0.0351860, -0.0685483, 0.0351860, -0.1037342, 0.1037342

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0552401
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0552486
time: 0.29 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0206600, 0.0216224, -0.0213640, 0.0238672, -0.0445272, 0.0429864
1: -0.0226827, 0.0442280, -0.0256137, 0.0480500, -0.0707327, 0.0698417
2: -0.0535189, 0.0294451, -0.0530172, 0.0317309, -0.0852498, 0.0824623
3: -0.0368305, 0.0571968, -0.0398649, 0.0629934, -0.0998239, 0.0970616
4: -0.0685483, 0.0351860, -0.0719933, 0.0355104, -0.1040587, 0.1071792

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0552401
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0552486
time: 0.31 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0201634, 0.0225399, -0.0200192, 0.0215588, -0.0417222, 0.0425591
1: -0.0234442, 0.0439457, -0.0220961, 0.0411768, -0.0646210, 0.0660418
2: -0.0503386, 0.0295319, -0.0501842, 0.0277946, -0.0781332, 0.0797161
3: -0.0375014, 0.0569067, -0.0360950, 0.0528612, -0.0903626, 0.0930016
4: -0.0680718, 0.0330660, -0.0649615, 0.0323495, -0.1004212, 0.0980275

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0551387
time: 0.31 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0551956
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0204120, 0.0225449, -0.0313514, 0.0346807, -0.0550927, 0.0538964
1: -0.0236487, 0.0436984, -0.0411882, 0.0823566, -0.1060053, 0.0848867
2: -0.0500013, 0.0294407, -0.0775521, 0.0542020, -0.1042033, 0.1069927
3: -0.0375547, 0.0566226, -0.0578171, 0.1109901, -0.1485448, 0.1144397
4: -0.0667728, 0.0328598, -0.1076526, 0.0603157, -0.1270885, 0.1405123

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551956, upper bound: 0.0551508
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551956, upper bound: 0.0552439
time: 0.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.79 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0552401
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0552486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0552401
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0552486
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0551387
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0551956
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0551956, upper bound: 0.0551508
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0551956, upper bound: 0.0552439

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0188579, 0.0197902, -0.0372305, 0.0372460
1: -0.0179365, 0.0346897, -0.0199836, 0.0387801, -0.0567166, 0.0546733
2: -0.0468226, 0.0235523, -0.0497252, 0.0260660, -0.0728886, 0.0732775
3: -0.0312331, 0.0437208, -0.0336468, 0.0494548, -0.0806879, 0.0773677
4: -0.0597628, 0.0287614, -0.0634654, 0.0315365, -0.0912994, 0.0922268

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552827, upper bound: 0.0551953
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552771, upper bound: 0.0552882
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0193022, 0.0197992, -0.0475980, 0.0509178
1: -0.0329395, 0.0745442, -0.0203114, 0.0382589, -0.0711985, 0.0948557
2: -0.0723793, 0.0529296, -0.0493214, 0.0263465, -0.0987258, 0.1022510
3: -0.0475005, 0.0975785, -0.0333894, 0.0487583, -0.0962589, 0.1309679
4: -0.0997654, 0.0589750, -0.0617646, 0.0314786, -0.1312440, 0.1207396

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552827, upper bound: 0.0551611
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552771, upper bound: 0.0552771
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0201634, 0.0225399, -0.0399801, 0.0385515
1: -0.0179365, 0.0346897, -0.0234442, 0.0439457, -0.0618822, 0.0581339
2: -0.0468226, 0.0235523, -0.0503386, 0.0295319, -0.0763545, 0.0738909
3: -0.0312331, 0.0437208, -0.0375014, 0.0569067, -0.0881398, 0.0812222
4: -0.0597628, 0.0287614, -0.0680718, 0.0330660, -0.0928288, 0.0968332

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550796, upper bound: 0.0552212
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550796, upper bound: 0.0552401
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0204120, 0.0225449, -0.0503437, 0.0520276
1: -0.0329395, 0.0745442, -0.0236487, 0.0436984, -0.0766380, 0.0981929
2: -0.0723793, 0.0529296, -0.0500013, 0.0294407, -0.1018200, 0.1029309
3: -0.0475005, 0.0975785, -0.0375547, 0.0566226, -0.1041232, 0.1351333
4: -0.0997654, 0.0589750, -0.0667728, 0.0328598, -0.1326252, 0.1257478

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550796, upper bound: 0.0552212
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550796, upper bound: 0.0552486
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0200192, 0.0215588, -0.0465115, 0.0489633
1: -0.0302124, 0.0647534, -0.0220961, 0.0411768, -0.0713892, 0.0868496
2: -0.0640738, 0.0455330, -0.0501842, 0.0277946, -0.0918684, 0.0957172
3: -0.0429836, 0.0844141, -0.0360950, 0.0528612, -0.0958448, 0.1205091
4: -0.0914269, 0.0490248, -0.0649615, 0.0323495, -0.1237764, 0.1139863

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0551956
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0551956
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0313514, 0.0346807, -0.0540157, 0.0529729
1: -0.0221682, 0.0413110, -0.0411882, 0.0823566, -0.1045248, 0.0824993
2: -0.0484735, 0.0279975, -0.0775521, 0.0542020, -0.1026755, 0.1055496
3: -0.0360067, 0.0530757, -0.0578171, 0.1109901, -0.1469968, 0.1108928
4: -0.0654610, 0.0313325, -0.1076526, 0.0603157, -0.1257767, 0.1389851

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0550796
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0550796
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0313514, 0.0346807, -0.0596334, 0.0602956
1: -0.0302124, 0.0647534, -0.0411882, 0.0823566, -0.1125689, 0.1059417
2: -0.0640738, 0.0455330, -0.0775521, 0.0542020, -0.1182758, 0.1230850
3: -0.0429836, 0.0844141, -0.0578171, 0.1109901, -0.1539737, 0.1422312
4: -0.0914269, 0.0490248, -0.1076526, 0.0603157, -0.1517426, 0.1566773

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550796, upper bound: 0.0552100
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0552100
time: 0.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.81 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.0552827, upper bound: 0.0551953
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.0552771, upper bound: 0.0552882
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.0552827, upper bound: 0.0551611
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.0552771, upper bound: 0.0552771
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.0550796, upper bound: 0.0552212
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.0550796, upper bound: 0.0552401
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.0550796, upper bound: 0.0552212
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.0550796, upper bound: 0.0552486
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0551956
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0551956
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0550796
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0550796
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.0550796, upper bound: 0.0552100
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0552100

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0170240, 0.0179375, -0.0187873, 0.0197516, -0.0367757, 0.0367248
1: -0.0174197, 0.0333798, -0.0196784, 0.0383622, -0.0557818, 0.0530582
2: -0.0457830, 0.0225719, -0.0495982, 0.0264650, -0.0722479, 0.0721700
3: -0.0305613, 0.0419471, -0.0331628, 0.0488801, -0.0794414, 0.0751099
4: -0.0583337, 0.0275973, -0.0637604, 0.0316531, -0.0899868, 0.0913577

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0551953
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0551953
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0174000, 0.0183441, -0.0183286, 0.0192303, -0.0366303, 0.0366726
1: -0.0178851, 0.0345685, -0.0192925, 0.0372437, -0.0551288, 0.0538610
2: -0.0467326, 0.0234628, -0.0485829, 0.0249485, -0.0716811, 0.0720457
3: -0.0311702, 0.0435545, -0.0327945, 0.0473565, -0.0785267, 0.0763490
4: -0.0596403, 0.0286617, -0.0619687, 0.0302832, -0.0899236, 0.0906305

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552882
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552882
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0274049, 0.0312060, -0.0191537, 0.0196810, -0.0470859, 0.0503597
1: -0.0323500, 0.0732470, -0.0199031, 0.0377920, -0.0701420, 0.0931501
2: -0.0714789, 0.0521259, -0.0491780, 0.0266379, -0.0981168, 0.1013039
3: -0.0467826, 0.0957815, -0.0327795, 0.0482967, -0.0950793, 0.1285609
4: -0.0985501, 0.0579949, -0.0622305, 0.0315074, -0.1300575, 0.1202254

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0551611
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0551611
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0277687, 0.0315840, -0.0188433, 0.0193236, -0.0470923, 0.0504274
1: -0.0328884, 0.0744569, -0.0197228, 0.0369795, -0.0698679, 0.0941797
2: -0.0723172, 0.0528730, -0.0483652, 0.0254182, -0.0977355, 0.1012383
3: -0.0474398, 0.0974495, -0.0326747, 0.0470361, -0.0944759, 0.1301242
4: -0.0996813, 0.0589098, -0.0604824, 0.0304306, -0.1301119, 0.1193922

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552771
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552771
time: 0.31 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0193350, 0.0216215, -0.0390617, 0.0377230
1: -0.0179365, 0.0346897, -0.0221682, 0.0413110, -0.0592475, 0.0568580
2: -0.0468226, 0.0235523, -0.0484735, 0.0279975, -0.0748202, 0.0720258
3: -0.0312331, 0.0437208, -0.0360067, 0.0530757, -0.0843088, 0.0797275
4: -0.0597628, 0.0287614, -0.0654610, 0.0313325, -0.0910953, 0.0942224

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544371, upper bound: 0.0537465
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551098, upper bound: 0.0552323
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0249527, 0.0289441, -0.0463844, 0.0433407
1: -0.0179365, 0.0346897, -0.0302124, 0.0647534, -0.0826899, 0.0649021
2: -0.0468226, 0.0235523, -0.0640738, 0.0455330, -0.0923556, 0.0876261
3: -0.0312331, 0.0437208, -0.0429836, 0.0844141, -0.1156472, 0.0867045
4: -0.0597628, 0.0287614, -0.0914269, 0.0490248, -0.1087876, 0.1201884

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544371, upper bound: 0.0552362
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551098, upper bound: 0.0552401
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0193350, 0.0216215, -0.0494203, 0.0509505
1: -0.0329395, 0.0745442, -0.0221682, 0.0413110, -0.0742506, 0.0967125
2: -0.0723793, 0.0529296, -0.0484735, 0.0279975, -0.1003769, 0.1014031
3: -0.0475005, 0.0975785, -0.0360067, 0.0530757, -0.1005763, 0.1335852
4: -0.0997654, 0.0589750, -0.0654610, 0.0313325, -0.1310980, 0.1244360

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544371, upper bound: 0.0537429
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550796, upper bound: 0.0552212
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0249527, 0.0289441, -0.0567429, 0.0565682
1: -0.0329395, 0.0745442, -0.0302124, 0.0647534, -0.0976930, 0.1047566
2: -0.0723793, 0.0529296, -0.0640738, 0.0455330, -0.1179123, 0.1170035
3: -0.0475005, 0.0975785, -0.0429836, 0.0844141, -0.1319146, 0.1405621
4: -0.0997654, 0.0589750, -0.0914269, 0.0490248, -0.1487902, 0.1504019

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544371, upper bound: 0.0552358
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550796, upper bound: 0.0552358
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0173510, 0.0182791, -0.0432318, 0.0462952
1: -0.0302124, 0.0647534, -0.0178209, 0.0343542, -0.0645666, 0.0825743
2: -0.0640738, 0.0455330, -0.0466004, 0.0233044, -0.0873782, 0.0921334
3: -0.0429836, 0.0844141, -0.0310903, 0.0432695, -0.0862531, 0.1155044
4: -0.0914269, 0.0490248, -0.0593948, 0.0285069, -0.1199338, 0.1084196

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543234, upper bound: 0.0536127
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551508, upper bound: 0.0551956
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0193350, 0.0216215, -0.0465741, 0.0482791
1: -0.0302124, 0.0647534, -0.0221682, 0.0413110, -0.0715234, 0.0869217
2: -0.0640738, 0.0455330, -0.0484735, 0.0279975, -0.0920714, 0.0940065
3: -0.0429836, 0.0844141, -0.0360067, 0.0530757, -0.0960593, 0.1204208
4: -0.0914269, 0.0490248, -0.0654610, 0.0313325, -0.1227595, 0.1144858

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543234, upper bound: 0.0536127
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551508, upper bound: 0.0551956
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0401096, 0.0435935, -0.0685462, 0.0690537
1: -0.0302124, 0.0647534, -0.0578089, 0.1086523, -0.1388647, 0.1225623
2: -0.0640738, 0.0455330, -0.0944850, 0.0669833, -0.1310571, 0.1400180
3: -0.0429836, 0.0844141, -0.0804732, 0.1510783, -0.1940619, 0.1648873
4: -0.0914269, 0.0490248, -0.1324382, 0.0749951, -0.1664220, 0.1814629

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551629, upper bound: 0.0552100
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552439, upper bound: 0.0552100
time: 0.32 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0236788, 0.0276065, -0.0525592, 0.0526229
1: -0.0302124, 0.0647534, -0.0270067, 0.0606095, -0.0908219, 0.0917602
2: -0.0640738, 0.0455330, -0.0618641, 0.0433749, -0.1074487, 0.1073970
3: -0.0429836, 0.0844141, -0.0392033, 0.0781467, -0.1211303, 0.1236174
4: -0.0914269, 0.0490248, -0.0880310, 0.0467448, -0.1381717, 0.1370558

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551629, upper bound: 0.0552100
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552439, upper bound: 0.0552100
time: 0.35 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.86 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0551953
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0551953
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552882
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552882
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0551611
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0551611
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552771
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552771
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0544371, upper bound: 0.0537465
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0551098, upper bound: 0.0552323
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0544371, upper bound: 0.0552362
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0551098, upper bound: 0.0552401
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0544371, upper bound: 0.0537429
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0550796, upper bound: 0.0552212
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0544371, upper bound: 0.0552358
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0550796, upper bound: 0.0552358
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0543234, upper bound: 0.0536127
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0551508, upper bound: 0.0551956
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0543234, upper bound: 0.0536127
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0551508, upper bound: 0.0551956
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0551629, upper bound: 0.0552100
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0552439, upper bound: 0.0552100
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0551629, upper bound: 0.0552100
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0552439, upper bound: 0.0552100

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0187873, 0.0197516, -0.0372363, 0.0372705
1: -0.0177545, 0.0346891, -0.0196784, 0.0383622, -0.0561167, 0.0543674
2: -0.0469266, 0.0241803, -0.0495982, 0.0264650, -0.0733916, 0.0737785
3: -0.0308810, 0.0437489, -0.0331628, 0.0488801, -0.0797611, 0.0769117
4: -0.0604094, 0.0290925, -0.0637604, 0.0316531, -0.0920625, 0.0928529

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552666, upper bound: 0.0551953
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552666, upper bound: 0.0551953
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0168805, 0.0177837, -0.0187873, 0.0197516, -0.0366321, 0.0365710
1: -0.0172204, 0.0330732, -0.0196784, 0.0383622, -0.0555826, 0.0527515
2: -0.0455946, 0.0223437, -0.0495982, 0.0264650, -0.0720596, 0.0719419
3: -0.0303572, 0.0414705, -0.0331628, 0.0488801, -0.0792373, 0.0746333
4: -0.0581362, 0.0274024, -0.0637604, 0.0316531, -0.0897893, 0.0911628

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552666, upper bound: 0.0551953
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552666, upper bound: 0.0551953
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0183286, 0.0192303, -0.0367149, 0.0368118
1: -0.0177545, 0.0346891, -0.0192925, 0.0372437, -0.0549983, 0.0539816
2: -0.0469266, 0.0241803, -0.0485829, 0.0249485, -0.0718751, 0.0727632
3: -0.0308810, 0.0437489, -0.0327945, 0.0473565, -0.0782375, 0.0765434
4: -0.0604094, 0.0290925, -0.0619687, 0.0302832, -0.0906926, 0.0910612

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552882
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552882
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0183286, 0.0192303, -0.0361440, 0.0361528
1: -0.0172633, 0.0331967, -0.0192925, 0.0372437, -0.0545071, 0.0524892
2: -0.0456765, 0.0224354, -0.0485829, 0.0249485, -0.0706250, 0.0710183
3: -0.0304100, 0.0416375, -0.0327945, 0.0473565, -0.0777665, 0.0744320
4: -0.0582723, 0.0274961, -0.0619687, 0.0302832, -0.0885555, 0.0894649

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552186
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552186
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0191537, 0.0196810, -0.0476898, 0.0516011
1: -0.0325964, 0.0775581, -0.0199031, 0.0377920, -0.0703885, 0.0974612
2: -0.0725141, 0.0540359, -0.0491780, 0.0266379, -0.0991520, 0.1032139
3: -0.0464773, 0.1009725, -0.0327795, 0.0482967, -0.0947739, 0.1337520
4: -0.1008744, 0.0594039, -0.0622305, 0.0315074, -0.1323818, 0.1216344

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552666, upper bound: 0.0551611
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552666, upper bound: 0.0551611
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0191537, 0.0196810, -0.0470188, 0.0502843
1: -0.0321509, 0.0732026, -0.0199031, 0.0377920, -0.0699430, 0.0931057
2: -0.0714296, 0.0520608, -0.0491780, 0.0266379, -0.0980675, 0.1012388
3: -0.0465936, 0.0955957, -0.0327795, 0.0482967, -0.0948902, 0.1283752
4: -0.0984768, 0.0579753, -0.0622305, 0.0315074, -0.1299842, 0.1202058

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552666, upper bound: 0.0551611
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552666, upper bound: 0.0551611
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0188433, 0.0193236, -0.0473324, 0.0512907
1: -0.0325964, 0.0775581, -0.0197228, 0.0369795, -0.0695759, 0.0972809
2: -0.0725141, 0.0540359, -0.0483652, 0.0254182, -0.0979323, 0.1024011
3: -0.0464773, 0.1009725, -0.0326747, 0.0470361, -0.0935134, 0.1336472
4: -0.1008744, 0.0594039, -0.0604824, 0.0304306, -0.1313050, 0.1198863

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552771
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552771
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0188433, 0.0193236, -0.0466614, 0.0499740
1: -0.0321509, 0.0732026, -0.0197228, 0.0369795, -0.0691304, 0.0929253
2: -0.0714296, 0.0520608, -0.0483652, 0.0254182, -0.0968479, 0.1004260
3: -0.0465936, 0.0955957, -0.0326747, 0.0470361, -0.0936297, 0.1282704
4: -0.0984768, 0.0579753, -0.0604824, 0.0304306, -0.1289074, 0.1184578

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0551611
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0551611
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0192832, 0.0215675, -0.0384811, 0.0371075
1: -0.0172633, 0.0331967, -0.0220989, 0.0411720, -0.0584353, 0.0552956
2: -0.0456765, 0.0224354, -0.0483697, 0.0279175, -0.0735941, 0.0708051
3: -0.0304100, 0.0416375, -0.0359199, 0.0528786, -0.0832886, 0.0775574
4: -0.0582723, 0.0274961, -0.0653247, 0.0312356, -0.0895079, 0.0928208

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 7

Time for candidate selection: 2.75 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535715, upper bound: 0.0544621
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548824, upper bound: 0.0550036
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0246403, 0.0286300, -0.0461146, 0.0431235
1: -0.0177545, 0.0346891, -0.0298407, 0.0637970, -0.0815516, 0.0645298
2: -0.0469266, 0.0241803, -0.0632847, 0.0448581, -0.0917848, 0.0874650
3: -0.0308810, 0.0437489, -0.0425067, 0.0830832, -0.1139641, 0.0862556
4: -0.0604094, 0.0290925, -0.0903697, 0.0481623, -0.1085716, 0.1194621

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551006, upper bound: 0.0551722
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551006, upper bound: 0.0552362
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0249210, 0.0289080, -0.0458216, 0.0427453
1: -0.0172633, 0.0331967, -0.0301650, 0.0646626, -0.0819259, 0.0633618
2: -0.0456765, 0.0224354, -0.0640108, 0.0454767, -0.0911532, 0.0864462
3: -0.0304100, 0.0416375, -0.0429279, 0.0842807, -0.1146907, 0.0845654
4: -0.0582723, 0.0274961, -0.0913416, 0.0489610, -0.1072333, 0.1188377

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0551722
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0552401
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0192832, 0.0215675, -0.0489053, 0.0504139
1: -0.0321509, 0.0732026, -0.0220989, 0.0411720, -0.0733229, 0.0953014
2: -0.0714296, 0.0520608, -0.0483697, 0.0279175, -0.0993472, 0.1004305
3: -0.0465936, 0.0955957, -0.0359199, 0.0528786, -0.0994721, 0.1315156
4: -0.0984768, 0.0579753, -0.0653247, 0.0312356, -0.1297124, 0.1233000

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 7

Time for candidate selection: 2.76 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533556, upper bound: 0.0543093
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548139, upper bound: 0.0549945
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0246403, 0.0286300, -0.0566388, 0.0570877
1: -0.0325964, 0.0775581, -0.0298407, 0.0637970, -0.0963935, 0.1073989
2: -0.0725141, 0.0540359, -0.0632847, 0.0448581, -0.1173722, 0.1173206
3: -0.0464773, 0.1009725, -0.0425067, 0.0830832, -0.1295604, 0.1434792
4: -0.1008744, 0.0594039, -0.0903697, 0.0481623, -0.1490366, 0.1497736

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550908, upper bound: 0.0550524
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550908, upper bound: 0.0552358
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0249210, 0.0289080, -0.0562459, 0.0560517
1: -0.0321509, 0.0732026, -0.0301650, 0.0646626, -0.0968135, 0.1033676
2: -0.0714296, 0.0520608, -0.0640108, 0.0454767, -0.1169063, 0.1160716
3: -0.0465936, 0.0955957, -0.0429279, 0.0842807, -0.1308743, 0.1385236
4: -0.0984768, 0.0579753, -0.0913416, 0.0489610, -0.1474378, 0.1493169

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551480, upper bound: 0.0550524
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551480, upper bound: 0.0552358
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0173206, 0.0182471, -0.0427283, 0.0457167
1: -0.0294868, 0.0633977, -0.0177822, 0.0342699, -0.0637567, 0.0811799
2: -0.0631345, 0.0446851, -0.0465351, 0.0232421, -0.0863766, 0.0912202
3: -0.0421317, 0.0824423, -0.0310431, 0.0431533, -0.0852849, 0.1134853
4: -0.0901240, 0.0480672, -0.0593130, 0.0284352, -0.1185591, 0.1073802

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552362, upper bound: 0.0551006
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552362, upper bound: 0.0552156
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0192832, 0.0215675, -0.0460486, 0.0476793
1: -0.0294868, 0.0633977, -0.0220989, 0.0411720, -0.0706587, 0.0854965
2: -0.0631345, 0.0446851, -0.0483697, 0.0279175, -0.0910520, 0.0930548
3: -0.0421317, 0.0824423, -0.0359199, 0.0528786, -0.0950102, 0.1183621
4: -0.0901240, 0.0480672, -0.0653247, 0.0312356, -0.1213596, 0.1133918

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 7

Time for candidate selection: 2.79 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536329, upper bound: 0.0544302
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549210, upper bound: 0.0549707
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0396337, 0.0431182, -0.0688808, 0.0701803
1: -0.0317584, 0.0698348, -0.0569457, 0.1070779, -0.1388363, 0.1267805
2: -0.0655043, 0.0478268, -0.0934623, 0.0660427, -0.1315470, 0.1412891
3: -0.0441914, 0.0912354, -0.0793711, 0.1488163, -0.1930077, 0.1706066
4: -0.0945988, 0.0507568, -0.1309454, 0.0738985, -0.1684973, 0.1817023

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551722, upper bound: 0.0551006
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551722, upper bound: 0.0552156
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0400793, 0.0435637, -0.0680449, 0.0684754
1: -0.0294868, 0.0633977, -0.0577613, 0.1085634, -0.1380502, 0.1211590
2: -0.0631345, 0.0446851, -0.0944216, 0.0669322, -0.1300667, 0.1391068
3: -0.0421317, 0.0824423, -0.0804142, 0.1509478, -0.1930795, 0.1628564
4: -0.0901240, 0.0480672, -0.1323537, 0.0749355, -0.1650595, 0.1804209

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552486, upper bound: 0.0551006
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552486, upper bound: 0.0552156
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0233747, 0.0273033, -0.0530659, 0.0539213
1: -0.0317584, 0.0698348, -0.0266454, 0.0596834, -0.0914417, 0.0964803
2: -0.0655043, 0.0478268, -0.0610874, 0.0427114, -0.1082157, 0.1089142
3: -0.0441914, 0.0912354, -0.0387396, 0.0768645, -0.1210559, 0.1299750
4: -0.0945988, 0.0507568, -0.0869918, 0.0458925, -0.1404913, 0.1377487

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551629, upper bound: 0.0550768
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551629, upper bound: 0.0552100
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0236513, 0.0275739, -0.0520551, 0.0520474
1: -0.0294868, 0.0633977, -0.0269659, 0.0605330, -0.0900198, 0.0903636
2: -0.0631345, 0.0446851, -0.0618095, 0.0433242, -0.1064587, 0.1064946
3: -0.0421317, 0.0824423, -0.0391558, 0.0780401, -0.1201717, 0.1215980
4: -0.0901240, 0.0480672, -0.0879554, 0.0466875, -0.1368114, 0.1360226

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552439, upper bound: 0.0550768
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552439, upper bound: 0.0552100
time: 0.34 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.69 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0552666, upper bound: 0.0551953
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0552666, upper bound: 0.0551953
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0552666, upper bound: 0.0551953
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0552666, upper bound: 0.0551953
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552882
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552882
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552186
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552186
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0552666, upper bound: 0.0551611
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0552666, upper bound: 0.0551611
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0552666, upper bound: 0.0551611
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0552666, upper bound: 0.0551611
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552771
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0552771
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0551611
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0551611, upper bound: 0.0551611
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0535715, upper bound: 0.0544621
IS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0548824, upper bound: 0.0550036
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0551006, upper bound: 0.0551722
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0551006, upper bound: 0.0552362
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0551722
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0552401
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0533556, upper bound: 0.0543093
IS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0548139, upper bound: 0.0549945
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0550908, upper bound: 0.0550524
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0550908, upper bound: 0.0552358
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0551480, upper bound: 0.0550524
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0551480, upper bound: 0.0552358
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0552362, upper bound: 0.0551006
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0552362, upper bound: 0.0552156
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0536329, upper bound: 0.0544302
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0549210, upper bound: 0.0549707
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0551722, upper bound: 0.0551006
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0551722, upper bound: 0.0552156
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0552486, upper bound: 0.0551006
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0552486, upper bound: 0.0552156
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0551629, upper bound: 0.0550768
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0551629, upper bound: 0.0552100
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0552439, upper bound: 0.0550768
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0552439, upper bound: 0.0552100

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0174846, 0.0184832, -0.0359678, 0.0359678
1: -0.0177545, 0.0346891, -0.0177545, 0.0346891, -0.0524436, 0.0524436
2: -0.0469266, 0.0241803, -0.0469266, 0.0241803, -0.0711069, 0.0711069
3: -0.0308810, 0.0437489, -0.0308810, 0.0437489, -0.0746299, 0.0746299
4: -0.0604094, 0.0290925, -0.0604094, 0.0290925, -0.0895019, 0.0895019

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0551503
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544429
time: 0.29 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0278479, 0.0313147, -0.0487994, 0.0463312
1: -0.0177545, 0.0346891, -0.0320241, 0.0745529, -0.0923074, 0.0667131
2: -0.0469266, 0.0241803, -0.0715611, 0.0517671, -0.0986937, 0.0957414
3: -0.0308810, 0.0437489, -0.0457391, 0.0967711, -0.1276521, 0.0894880
4: -0.0604094, 0.0290925, -0.0984275, 0.0572647, -0.1176741, 0.1275199

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0551715
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544692
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0168805, 0.0177837, -0.0174846, 0.0184832, -0.0353637, 0.0352684
1: -0.0172204, 0.0330732, -0.0177545, 0.0346891, -0.0519095, 0.0508277
2: -0.0455946, 0.0223437, -0.0469266, 0.0241803, -0.0697750, 0.0692703
3: -0.0303572, 0.0414705, -0.0308810, 0.0437489, -0.0741061, 0.0723515
4: -0.0581362, 0.0274024, -0.0604094, 0.0290925, -0.0872287, 0.0878118

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546611, upper bound: 0.0551424
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546988, upper bound: 0.0549208
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0168805, 0.0177837, -0.0278479, 0.0313147, -0.0481952, 0.0456317
1: -0.0172204, 0.0330732, -0.0320241, 0.0745529, -0.0917733, 0.0650972
2: -0.0455946, 0.0223437, -0.0715611, 0.0517671, -0.0973618, 0.0939048
3: -0.0303572, 0.0414705, -0.0457391, 0.0967711, -0.1271283, 0.0872096
4: -0.0581362, 0.0274024, -0.0984275, 0.0572647, -0.1154010, 0.1258298

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546611, upper bound: 0.0551424
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546988, upper bound: 0.0549470
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0169136, 0.0178243, -0.0353089, 0.0353969
1: -0.0177545, 0.0346891, -0.0172633, 0.0331967, -0.0509512, 0.0519524
2: -0.0469266, 0.0241803, -0.0456765, 0.0224354, -0.0693620, 0.0698568
3: -0.0308810, 0.0437489, -0.0304100, 0.0416375, -0.0725185, 0.0741589
4: -0.0604094, 0.0290925, -0.0582723, 0.0274961, -0.0879055, 0.0873647

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549491, upper bound: 0.0552778
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549208, upper bound: 0.0546988
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0272046, 0.0301253, -0.0476099, 0.0456878
1: -0.0177545, 0.0346891, -0.0316691, 0.0706761, -0.0884306, 0.0663582
2: -0.0469266, 0.0241803, -0.0705942, 0.0499720, -0.0968986, 0.0947745
3: -0.0308810, 0.0437489, -0.0459514, 0.0921504, -0.1230314, 0.0897003
4: -0.0604094, 0.0290925, -0.0963448, 0.0559998, -0.1164092, 0.1254373

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549491, upper bound: 0.0552778
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549208, upper bound: 0.0546988
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0169136, 0.0178243, -0.0347379, 0.0347379
1: -0.0172633, 0.0331967, -0.0172633, 0.0331967, -0.0504600, 0.0504600
2: -0.0456765, 0.0224354, -0.0456765, 0.0224354, -0.0681119, 0.0681119
3: -0.0304100, 0.0416375, -0.0304100, 0.0416375, -0.0720475, 0.0720475
4: -0.0582723, 0.0274961, -0.0582723, 0.0274961, -0.0857684, 0.0857684

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544236, upper bound: 0.0534880
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552548, upper bound: 0.0551985
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0272046, 0.0301253, -0.0470389, 0.0450289
1: -0.0172633, 0.0331967, -0.0316691, 0.0706761, -0.0879394, 0.0648659
2: -0.0456765, 0.0224354, -0.0705942, 0.0499720, -0.0956485, 0.0930296
3: -0.0304100, 0.0416375, -0.0459514, 0.0921504, -0.1225604, 0.0875890
4: -0.0582723, 0.0274961, -0.0963448, 0.0559998, -0.1142721, 0.1238410

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544236, upper bound: 0.0534880
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552548, upper bound: 0.0551985
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0174846, 0.0184832, -0.0464920, 0.0499320
1: -0.0325964, 0.0775581, -0.0177545, 0.0346891, -0.0672855, 0.0953127
2: -0.0725141, 0.0540359, -0.0469266, 0.0241803, -0.0966944, 0.1009625
3: -0.0464773, 0.1009725, -0.0308810, 0.0437489, -0.0902262, 0.1318535
4: -0.1008744, 0.0594039, -0.0604094, 0.0290925, -0.1299669, 0.1198133

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535074, upper bound: 0.0528424
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0552530
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0278479, 0.0313147, -0.0593235, 0.0602953
1: -0.0325964, 0.0775581, -0.0320241, 0.0745529, -0.1071493, 0.1095822
2: -0.0725141, 0.0540359, -0.0715611, 0.0517671, -0.1242812, 0.1255970
3: -0.0464773, 0.1009725, -0.0457391, 0.0967711, -0.1432484, 0.1467116
4: -0.1008744, 0.0594039, -0.0984275, 0.0572647, -0.1581391, 0.1578314

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535074, upper bound: 0.0528938
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0552530
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0174846, 0.0184832, -0.0458211, 0.0486153
1: -0.0321509, 0.0732026, -0.0177545, 0.0346891, -0.0668400, 0.0909571
2: -0.0714296, 0.0520608, -0.0469266, 0.0241803, -0.0956100, 0.0989874
3: -0.0465936, 0.0955957, -0.0308810, 0.0437489, -0.0903425, 0.1264767
4: -0.0984768, 0.0579753, -0.0604094, 0.0290925, -0.1275693, 0.1183847

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534492, upper bound: 0.0528129
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552657, upper bound: 0.0551330
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0278479, 0.0313147, -0.0586526, 0.0589786
1: -0.0321509, 0.0732026, -0.0320241, 0.0745529, -0.1067038, 0.1052266
2: -0.0714296, 0.0520608, -0.0715611, 0.0517671, -0.1231968, 0.1236219
3: -0.0465936, 0.0955957, -0.0457391, 0.0967711, -0.1433647, 0.1413348
4: -0.0984768, 0.0579753, -0.0984275, 0.0572647, -0.1557416, 0.1564028

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534492, upper bound: 0.0528938
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552657, upper bound: 0.0551330
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0168805, 0.0177837, -0.0457926, 0.0493279
1: -0.0325964, 0.0775581, -0.0172204, 0.0330732, -0.0656696, 0.0947786
2: -0.0725141, 0.0540359, -0.0455946, 0.0223437, -0.0948578, 0.0996305
3: -0.0464773, 0.1009725, -0.0303572, 0.0414705, -0.0879478, 0.1313297
4: -0.1008744, 0.0594039, -0.0581362, 0.0274024, -0.1282768, 0.1175401

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546706, upper bound: 0.0533062
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551330, upper bound: 0.0552657
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0272046, 0.0301253, -0.0581341, 0.0596520
1: -0.0325964, 0.0775581, -0.0316691, 0.0706761, -0.1032725, 0.1092273
2: -0.0725141, 0.0540359, -0.0705942, 0.0499720, -0.1224861, 0.1246301
3: -0.0464773, 0.1009725, -0.0459514, 0.0921504, -0.1386277, 0.1469239
4: -0.1008744, 0.0594039, -0.0963448, 0.0559998, -0.1568742, 0.1557487

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546706, upper bound: 0.0533062
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551330, upper bound: 0.0552657
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0168805, 0.0177837, -0.0451216, 0.0480111
1: -0.0321509, 0.0732026, -0.0172204, 0.0330732, -0.0652241, 0.0904230
2: -0.0714296, 0.0520608, -0.0455946, 0.0223437, -0.0937734, 0.0976554
3: -0.0465936, 0.0955957, -0.0303572, 0.0414705, -0.0880641, 0.1259529
4: -0.0984768, 0.0579753, -0.0581362, 0.0274024, -0.1258792, 0.1161115

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544662, upper bound: 0.0535078
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552548, upper bound: 0.0551330
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0272046, 0.0301253, -0.0574632, 0.0583352
1: -0.0321509, 0.0732026, -0.0316691, 0.0706761, -0.1028270, 0.1048717
2: -0.0714296, 0.0520608, -0.0705942, 0.0499720, -0.1214016, 0.1226550
3: -0.0465936, 0.0955957, -0.0459514, 0.0921504, -0.1387440, 0.1415471
4: -0.0984768, 0.0579753, -0.0963448, 0.0559998, -0.1544766, 0.1543202

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544662, upper bound: 0.0535078
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552548, upper bound: 0.0551330
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0257626, 0.0305466, -0.0480312, 0.0442459
1: -0.0177545, 0.0346891, -0.0317584, 0.0698348, -0.0875894, 0.0664474
2: -0.0469266, 0.0241803, -0.0655043, 0.0478268, -0.0947534, 0.0896846
3: -0.0308810, 0.0437489, -0.0441914, 0.0912354, -0.1221164, 0.0879403
4: -0.0604094, 0.0290925, -0.0945988, 0.0507568, -0.1111662, 0.1236912

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549722, upper bound: 0.0551782
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549439, upper bound: 0.0545446
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0174846, 0.0184832, -0.0244811, 0.0283961, -0.0458807, 0.0429644
1: -0.0177545, 0.0346891, -0.0294868, 0.0633977, -0.0811522, 0.0641759
2: -0.0469266, 0.0241803, -0.0631345, 0.0446851, -0.0916117, 0.0873148
3: -0.0308810, 0.0437489, -0.0421317, 0.0824423, -0.1133233, 0.0858805
4: -0.0604094, 0.0290925, -0.0901240, 0.0480672, -0.1084765, 0.1192164

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549722, upper bound: 0.0552178
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549439, upper bound: 0.0546783
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0257626, 0.0305466, -0.0474602, 0.0435869
1: -0.0172633, 0.0331967, -0.0317584, 0.0698348, -0.0870982, 0.0649551
2: -0.0456765, 0.0224354, -0.0655043, 0.0478268, -0.0935033, 0.0879397
3: -0.0304100, 0.0416375, -0.0441914, 0.0912354, -0.1216454, 0.0858289
4: -0.0582723, 0.0274961, -0.0945988, 0.0507568, -0.1090291, 0.1220949

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550871, upper bound: 0.0551628
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550907, upper bound: 0.0550398
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0169136, 0.0178243, -0.0244811, 0.0283961, -0.0453097, 0.0423054
1: -0.0172633, 0.0331967, -0.0294868, 0.0633977, -0.0806610, 0.0626835
2: -0.0456765, 0.0224354, -0.0631345, 0.0446851, -0.0903616, 0.0855699
3: -0.0304100, 0.0416375, -0.0421317, 0.0824423, -0.1128523, 0.0837692
4: -0.0582723, 0.0274961, -0.0901240, 0.0480672, -0.1063394, 0.1176201

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550871, upper bound: 0.0551628
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550907, upper bound: 0.0551146
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0280088, 0.0324474, -0.0244811, 0.0283961, -0.0564049, 0.0569285
1: -0.0325964, 0.0775581, -0.0294868, 0.0633977, -0.0959941, 0.1070449
2: -0.0725141, 0.0540359, -0.0631345, 0.0446851, -0.1171992, 0.1171703
3: -0.0464773, 0.1009725, -0.0421317, 0.0824423, -0.1289196, 0.1431042
4: -0.1008744, 0.0594039, -0.0901240, 0.0480672, -0.1489415, 0.1495278

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544191, upper bound: 0.0529508
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550608, upper bound: 0.0552126
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0273379, 0.0311306, -0.0244811, 0.0283961, -0.0557340, 0.0556118
1: -0.0321509, 0.0732026, -0.0294868, 0.0633977, -0.0955486, 0.1026894
2: -0.0714296, 0.0520608, -0.0631345, 0.0446851, -0.1161148, 0.1151953
3: -0.0465936, 0.0955957, -0.0421317, 0.0824423, -0.1290358, 0.1377274
4: -0.0984768, 0.0579753, -0.0901240, 0.0480672, -0.1465440, 0.1480993

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543258, upper bound: 0.0535741
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551071, upper bound: 0.0550280
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0174846, 0.0184832, -0.0429644, 0.0458807
1: -0.0294868, 0.0633977, -0.0177545, 0.0346891, -0.0641759, 0.0811522
2: -0.0631345, 0.0446851, -0.0469266, 0.0241803, -0.0873148, 0.0916117
3: -0.0421317, 0.0824423, -0.0308810, 0.0437489, -0.0858805, 0.1133233
4: -0.0901240, 0.0480672, -0.0604094, 0.0290925, -0.1192164, 0.1084765

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534312, upper bound: 0.0528129
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552138, upper bound: 0.0550707
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0168805, 0.0177837, -0.0422649, 0.0452766
1: -0.0294868, 0.0633977, -0.0172204, 0.0330732, -0.0625599, 0.0806181
2: -0.0631345, 0.0446851, -0.0455946, 0.0223437, -0.0854782, 0.0902798
3: -0.0421317, 0.0824423, -0.0303572, 0.0414705, -0.0836021, 0.1127995
4: -0.0901240, 0.0480672, -0.0581362, 0.0274024, -0.1175263, 0.1062034

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534312, upper bound: 0.0535492
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552138, upper bound: 0.0550707
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0409758, 0.0450107, -0.0707733, 0.0715224
1: -0.0317584, 0.0698348, -0.0604399, 0.1128874, -0.1446458, 0.1302747
2: -0.0655043, 0.0478268, -0.0957154, 0.0688422, -0.1343465, 0.1435421
3: -0.0441914, 0.0912354, -0.0840132, 0.1566700, -0.2008614, 0.1752486
4: -0.0945988, 0.0507568, -0.1348560, 0.0763531, -0.1709519, 0.1856128

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546567, upper bound: 0.0529284
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551501, upper bound: 0.0551900
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0396456, 0.0431375, -0.0689002, 0.0701921
1: -0.0317584, 0.0698348, -0.0570782, 0.1072880, -0.1390464, 0.1269130
2: -0.0655043, 0.0478268, -0.0935175, 0.0662005, -0.1317048, 0.1413443
3: -0.0441914, 0.0912354, -0.0795728, 0.1490739, -0.1932652, 0.1708082
4: -0.0945988, 0.0507568, -0.1311446, 0.0740834, -0.1686822, 0.1819014

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546567, upper bound: 0.0533986
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551501, upper bound: 0.0551920
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0409758, 0.0450107, -0.0694918, 0.0693719
1: -0.0294868, 0.0633977, -0.0604399, 0.1128874, -0.1423742, 0.1238375
2: -0.0631345, 0.0446851, -0.0957154, 0.0688422, -0.1319767, 0.1404005
3: -0.0421317, 0.0824423, -0.0840132, 0.1566700, -0.1988016, 0.1664554
4: -0.0901240, 0.0480672, -0.1348560, 0.0763531, -0.1664771, 0.1829232

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542673, upper bound: 0.0529278
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552257, upper bound: 0.0550707
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0396456, 0.0431375, -0.0676187, 0.0680417
1: -0.0294868, 0.0633977, -0.0570782, 0.1072880, -0.1367748, 0.1204759
2: -0.0631345, 0.0446851, -0.0935175, 0.0662005, -0.1293350, 0.1382026
3: -0.0421317, 0.0824423, -0.0795728, 0.1490739, -0.1912055, 0.1620151
4: -0.0901240, 0.0480672, -0.1311446, 0.0740834, -0.1642074, 0.1792117

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542673, upper bound: 0.0535255
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552257, upper bound: 0.0550707
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0240311, 0.0286206, -0.0543832, 0.0545777
1: -0.0317584, 0.0698348, -0.0273552, 0.0640706, -0.0958289, 0.0971900
2: -0.0655043, 0.0478268, -0.0623516, 0.0446298, -0.1101340, 0.1101784
3: -0.0441914, 0.0912354, -0.0389960, 0.0823029, -0.1264943, 0.1302314
4: -0.0945988, 0.0507568, -0.0896403, 0.0474051, -0.1420038, 0.1403971

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546614, upper bound: 0.0535113
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551385, upper bound: 0.0551804
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0257626, 0.0305466, -0.0232538, 0.0271039, -0.0528665, 0.0538003
1: -0.0317584, 0.0698348, -0.0263714, 0.0594278, -0.0911861, 0.0962062
2: -0.0655043, 0.0478268, -0.0610230, 0.0425949, -0.1080991, 0.1088498
3: -0.0441914, 0.0912354, -0.0384619, 0.0764992, -0.1206906, 0.1296974
4: -0.0945988, 0.0507568, -0.0868614, 0.0458645, -0.1404632, 0.1376182

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546614, upper bound: 0.0537025
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551385, upper bound: 0.0551863
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0240311, 0.0286206, -0.0531017, 0.0524272
1: -0.0294868, 0.0633977, -0.0273552, 0.0640706, -0.0935574, 0.0907529
2: -0.0631345, 0.0446851, -0.0623516, 0.0446298, -0.1077642, 0.1070367
3: -0.0421317, 0.0824423, -0.0389960, 0.0823029, -0.1244345, 0.1214383
4: -0.0901240, 0.0480672, -0.0896403, 0.0474051, -0.1375290, 0.1377074

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544632, upper bound: 0.0534960
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552197, upper bound: 0.0550493
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0244811, 0.0283961, -0.0232538, 0.0271039, -0.0515850, 0.0516499
1: -0.0294868, 0.0633977, -0.0263714, 0.0594278, -0.0889146, 0.0897691
2: -0.0631345, 0.0446851, -0.0610230, 0.0425949, -0.1057293, 0.1057081
3: -0.0421317, 0.0824423, -0.0384619, 0.0764992, -0.1186309, 0.1209042
4: -0.0901240, 0.0480672, -0.0868614, 0.0458645, -0.1359884, 0.1349286

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544632, upper bound: 0.0536402
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552197, upper bound: 0.0550493
time: 0.36 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.87 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0551503
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544429
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0551715
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544692
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0546611, upper bound: 0.0551424
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0546988, upper bound: 0.0549208
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0546611, upper bound: 0.0551424
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0546988, upper bound: 0.0549470
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0549491, upper bound: 0.0552778
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0549208, upper bound: 0.0546988
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0549491, upper bound: 0.0552778
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0549208, upper bound: 0.0546988
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0544236, upper bound: 0.0534880
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0552548, upper bound: 0.0551985
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0544236, upper bound: 0.0534880
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0552548, upper bound: 0.0551985
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0535074, upper bound: 0.0528424
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0552530
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0535074, upper bound: 0.0528938
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0552530
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0534492, upper bound: 0.0528129
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0552657, upper bound: 0.0551330
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0534492, upper bound: 0.0528938
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0552657, upper bound: 0.0551330
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0546706, upper bound: 0.0533062
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0551330, upper bound: 0.0552657
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0546706, upper bound: 0.0533062
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0551330, upper bound: 0.0552657
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0544662, upper bound: 0.0535078
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0552548, upper bound: 0.0551330
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0544662, upper bound: 0.0535078
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0552548, upper bound: 0.0551330
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0549722, upper bound: 0.0551782
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0549439, upper bound: 0.0545446
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0549722, upper bound: 0.0552178
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0549439, upper bound: 0.0546783
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0550871, upper bound: 0.0551628
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0550907, upper bound: 0.0550398
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0550871, upper bound: 0.0551628
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0550907, upper bound: 0.0551146
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0544191, upper bound: 0.0529508
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0550608, upper bound: 0.0552126
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0543258, upper bound: 0.0535741
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0551071, upper bound: 0.0550280
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0534312, upper bound: 0.0528129
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0552138, upper bound: 0.0550707
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0534312, upper bound: 0.0535492
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0552138, upper bound: 0.0550707
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0546567, upper bound: 0.0529284
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0551501, upper bound: 0.0551900
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0546567, upper bound: 0.0533986
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0551501, upper bound: 0.0551920
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0542673, upper bound: 0.0529278
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0552257, upper bound: 0.0550707
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0542673, upper bound: 0.0535255
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0552257, upper bound: 0.0550707
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0546614, upper bound: 0.0535113
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0551385, upper bound: 0.0551804
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0546614, upper bound: 0.0537025
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0551385, upper bound: 0.0551863
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0544632, upper bound: 0.0534960
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0552197, upper bound: 0.0550493
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0544632, upper bound: 0.0536402
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.87
Output dim: 0, lower bound: -0.0552197, upper bound: 0.0550493

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0276823, 0.0311695, -0.0500024, 0.0468001
1: -0.0185345, 0.0336902, -0.0315817, 0.0740637, -0.0925982, 0.0652719
2: -0.0465600, 0.0245035, -0.0712842, 0.0515271, -0.0980870, 0.0957877
3: -0.0314490, 0.0414548, -0.0452620, 0.0960127, -0.1274617, 0.0867168
4: -0.0564701, 0.0289175, -0.0980529, 0.0569954, -0.1134656, 0.1269704

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0528242, upper bound: 0.0534609
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548459, upper bound: 0.0551403
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0166276, 0.0174877, -0.0363205, 0.0357455
1: -0.0185345, 0.0336902, -0.0165990, 0.0320485, -0.0505831, 0.0502891
2: -0.0465600, 0.0245035, -0.0450906, 0.0219316, -0.0684916, 0.0695941
3: -0.0314490, 0.0414548, -0.0294627, 0.0400382, -0.0714873, 0.0709175
4: -0.0564701, 0.0289175, -0.0574879, 0.0269119, -0.0833820, 0.0864053

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549208, upper bound: 0.0546611
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549208, upper bound: 0.0546988
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0270240, 0.0299775, -0.0488103, 0.0461419
1: -0.0185345, 0.0336902, -0.0311674, 0.0700214, -0.0885559, 0.0648576
2: -0.0465600, 0.0245035, -0.0702907, 0.0497460, -0.0963060, 0.0947942
3: -0.0314490, 0.0414548, -0.0454295, 0.0910918, -0.1225408, 0.0868843
4: -0.0564701, 0.0289175, -0.0959161, 0.0557444, -0.1122146, 0.1248335

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549551, upper bound: 0.0545802
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549551, upper bound: 0.0546988
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0163001, 0.0171033, -0.0169136, 0.0178243, -0.0341244, 0.0340169
1: -0.0163531, 0.0312095, -0.0172633, 0.0331967, -0.0495499, 0.0484729
2: -0.0443946, 0.0211404, -0.0456765, 0.0224354, -0.0668300, 0.0668169
3: -0.0292436, 0.0388854, -0.0304100, 0.0416375, -0.0708811, 0.0692954
4: -0.0566717, 0.0260474, -0.0582723, 0.0274961, -0.0841679, 0.0843197

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536281, upper bound: 0.0547423
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0536281, upper bound: 0.0552000
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0163001, 0.0171033, -0.0272046, 0.0301253, -0.0464254, 0.0443079
1: -0.0163531, 0.0312095, -0.0316691, 0.0706761, -0.0870292, 0.0628787
2: -0.0443946, 0.0211404, -0.0705942, 0.0499720, -0.0943666, 0.0917346
3: -0.0292436, 0.0388854, -0.0459514, 0.0921504, -0.1213940, 0.0848368
4: -0.0566717, 0.0260474, -0.0963448, 0.0559998, -0.1126716, 0.1223922

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536489, upper bound: 0.0547490
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0536489, upper bound: 0.0551985
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0275401, 0.0318847, -0.0174846, 0.0184832, -0.0460233, 0.0493693
1: -0.0315660, 0.0759994, -0.0177545, 0.0346891, -0.0662550, 0.0937540
2: -0.0714210, 0.0530449, -0.0469266, 0.0241803, -0.0956013, 0.0999715
3: -0.0452617, 0.0986971, -0.0308810, 0.0437489, -0.0890106, 0.1295781
4: -0.0993945, 0.0582935, -0.0604094, 0.0290925, -0.1284870, 0.1187029

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551403, upper bound: 0.0548459
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543515, upper bound: 0.0548036
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0275401, 0.0318847, -0.0278479, 0.0313147, -0.0588548, 0.0597326
1: -0.0315660, 0.0759994, -0.0320241, 0.0745529, -0.1061188, 0.1080235
2: -0.0714210, 0.0530449, -0.0715611, 0.0517671, -0.1231881, 0.1246060
3: -0.0452617, 0.0986971, -0.0457391, 0.0967711, -0.1420329, 0.1444362
4: -0.0993945, 0.0582935, -0.0984275, 0.0572647, -0.1566593, 0.1567210

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0528446, upper bound: 0.0538617
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0528446, upper bound: 0.0552530
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0268415, 0.0305504, -0.0174846, 0.0184832, -0.0453247, 0.0480350
1: -0.0310880, 0.0714282, -0.0177545, 0.0346891, -0.0657771, 0.0891827
2: -0.0703224, 0.0510674, -0.0469266, 0.0241803, -0.0945028, 0.0979940
3: -0.0453250, 0.0927791, -0.0308810, 0.0437489, -0.0890739, 0.1236601
4: -0.0969780, 0.0568669, -0.0604094, 0.0290925, -0.1260705, 0.1172763

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552572, upper bound: 0.0549521
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545914, upper bound: 0.0549097
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0268415, 0.0305504, -0.0278479, 0.0313147, -0.0581562, 0.0583984
1: -0.0310880, 0.0714282, -0.0320241, 0.0745529, -0.1056409, 0.1034522
2: -0.0703224, 0.0510674, -0.0715611, 0.0517671, -0.1220896, 0.1226285
3: -0.0453250, 0.0927791, -0.0457391, 0.0967711, -0.1420961, 0.1385182
4: -0.0969780, 0.0568669, -0.0984275, 0.0572647, -0.1542428, 0.1552944

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529139, upper bound: 0.0537930
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529139, upper bound: 0.0551330
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0275401, 0.0318847, -0.0168805, 0.0177837, -0.0453238, 0.0487651
1: -0.0315660, 0.0759994, -0.0172204, 0.0330732, -0.0646391, 0.0932199
2: -0.0714210, 0.0530449, -0.0455946, 0.0223437, -0.0937647, 0.0986395
3: -0.0452617, 0.0986971, -0.0303572, 0.0414705, -0.0867322, 0.1290543
4: -0.0993945, 0.0582935, -0.0581362, 0.0274024, -0.1267969, 0.1164297

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551295, upper bound: 0.0549946
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549130, upper bound: 0.0550109
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0275401, 0.0318847, -0.0272046, 0.0301253, -0.0576654, 0.0590893
1: -0.0315660, 0.0759994, -0.0316691, 0.0706761, -0.1022421, 0.1076686
2: -0.0714210, 0.0530449, -0.0705942, 0.0499720, -0.1213930, 0.1236390
3: -0.0452617, 0.0986971, -0.0459514, 0.0921504, -0.1374122, 0.1446485
4: -0.0993945, 0.0582935, -0.0963448, 0.0559998, -0.1553943, 0.1546383

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0528701, upper bound: 0.0540368
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0528701, upper bound: 0.0552657
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0268415, 0.0305504, -0.0168805, 0.0177837, -0.0446252, 0.0474309
1: -0.0310880, 0.0714282, -0.0172204, 0.0330732, -0.0641612, 0.0886486
2: -0.0703224, 0.0510674, -0.0455946, 0.0223437, -0.0926662, 0.0966620
3: -0.0453250, 0.0927791, -0.0303572, 0.0414705, -0.0867955, 0.1231363
4: -0.0969780, 0.0568669, -0.0581362, 0.0274024, -0.1243804, 0.1150032

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534739, upper bound: 0.0547219
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534739, upper bound: 0.0547219
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0268415, 0.0305504, -0.0272046, 0.0301253, -0.0569668, 0.0577550
1: -0.0310880, 0.0714282, -0.0316691, 0.0706761, -0.1017641, 0.1030973
2: -0.0703224, 0.0510674, -0.0705942, 0.0499720, -0.1202944, 0.1216615
3: -0.0453250, 0.0927791, -0.0459514, 0.0921504, -0.1374754, 0.1387305
4: -0.0969780, 0.0568669, -0.0963448, 0.0559998, -0.1529779, 0.1532118

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534739, upper bound: 0.0547242
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534739, upper bound: 0.0551330
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0255518, 0.0303650, -0.0491978, 0.0446697
1: -0.0185345, 0.0336902, -0.0312596, 0.0691960, -0.0877305, 0.0649498
2: -0.0465600, 0.0245035, -0.0651739, 0.0475672, -0.0941272, 0.0896775
3: -0.0314490, 0.0414548, -0.0436242, 0.0902065, -0.1216555, 0.0850790
4: -0.0564701, 0.0289175, -0.0940844, 0.0504721, -0.1069422, 0.1230019

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0528356, upper bound: 0.0543758
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549449, upper bound: 0.0551576
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0188328, 0.0191179, -0.0242811, 0.0282284, -0.0470612, 0.0433989
1: -0.0185345, 0.0336902, -0.0290191, 0.0628083, -0.0813429, 0.0627093
2: -0.0465600, 0.0245035, -0.0628337, 0.0444479, -0.0910079, 0.0873372
3: -0.0314490, 0.0414548, -0.0416199, 0.0815022, -0.1129512, 0.0830747
4: -0.0564701, 0.0289175, -0.0896599, 0.0478091, -0.1042792, 0.1185774

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549551, upper bound: 0.0545035
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549551, upper bound: 0.0546783
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0183495, 0.0186593, -0.0255518, 0.0303650, -0.0487145, 0.0442111
1: -0.0180988, 0.0325174, -0.0312596, 0.0691960, -0.0872947, 0.0637771
2: -0.0456146, 0.0231816, -0.0651739, 0.0475672, -0.0931819, 0.0883556
3: -0.0310402, 0.0397046, -0.0436242, 0.0902065, -0.1212466, 0.0833288
4: -0.0545507, 0.0277470, -0.0940844, 0.0504721, -0.1050228, 0.1218314

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0531288, upper bound: 0.0546125
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550681, upper bound: 0.0551426
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0183495, 0.0186593, -0.0242811, 0.0282284, -0.0465779, 0.0429404
1: -0.0180988, 0.0325174, -0.0290191, 0.0628083, -0.0809071, 0.0615366
2: -0.0456146, 0.0231816, -0.0628337, 0.0444479, -0.0900626, 0.0860153
3: -0.0310402, 0.0397046, -0.0416199, 0.0815022, -0.1125424, 0.0813245
4: -0.0545507, 0.0277470, -0.0896599, 0.0478091, -0.1023598, 0.1174069

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0527683, upper bound: 0.0544120
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551238, upper bound: 0.0551426
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0275401, 0.0318847, -0.0244811, 0.0283961, -0.0559362, 0.0563658
1: -0.0315660, 0.0759994, -0.0294868, 0.0633977, -0.0949637, 0.1054862
2: -0.0714210, 0.0530449, -0.0631345, 0.0446851, -0.1161061, 0.1161793
3: -0.0452617, 0.0986971, -0.0421317, 0.0824423, -0.1277040, 0.1408287
4: -0.0993945, 0.0582935, -0.0901240, 0.0480672, -0.1474617, 0.1484174

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0528701, upper bound: 0.0537574
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0528701, upper bound: 0.0537574
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0238435, 0.0277011, -0.0174846, 0.0184832, -0.0423268, 0.0451857
1: -0.0284588, 0.0615482, -0.0177545, 0.0346891, -0.0631479, 0.0793027
2: -0.0619252, 0.0435389, -0.0469266, 0.0241803, -0.0861056, 0.0904655
3: -0.0407595, 0.0795807, -0.0308810, 0.0437489, -0.0845083, 0.1104617
4: -0.0884067, 0.0467996, -0.0604094, 0.0290925, -0.1174992, 0.1072090

Time for backsubstitution: 2.02 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0036636, high=0.0282057, mid=0.0282057, abs_max=0.058847926557064056
rel_dist={0: [-0.055451745298252274, 0.05545174529825231]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.003663635035536572
execution time: 1148.17 seconds
