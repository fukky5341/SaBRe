## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.1823463684


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.5293298, 2.5293295)
1: (-10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2577815, 2.2577815)
2: (-6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.3718305, 2.3718295)
3: (-2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8441834, 1.8441832)
4: (-6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1593237, 3.1593237)
5: (-8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4321971, 2.4321966)
6: (-19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1931105, 3.1931105)
7: (4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772)
8: (-7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3909245, 2.3909245)
9: (-7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6847959, 2.6847959)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.35 + 34.35 = 57.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -1.1847179, upper bound: 1.1847154

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 457

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1841504, upper bound: 1.1791575
time: 7.22 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847100, upper bound: 1.1847084
time: 4.40 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 11.70 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 11.70
Output dim: 7, lower bound: -1.1841504, upper bound: 1.1791575
NS_A2, status: Status.UNKNOWN, split count: 1, time: 11.70
Output dim: 7, lower bound: -1.1847100, upper bound: 1.1847084

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -17.5882187, -13.5900822, -17.5930214, -13.5878315, -2.5169849, 2.5178177
1: -10.2623758, -7.4767728, -10.2640038, -7.4714136, -2.2490282, 2.2455969
2: -6.4378533, -3.5996528, -6.4474549, -3.5983841, -2.3524609, 2.3608422
3: -2.4340112, 0.1182401, -2.4360194, 0.1221886, -1.8353758, 1.8326530
4: -6.9883199, -2.9186769, -6.9913054, -2.9069707, -3.1420994, 3.1336880
5: -8.9537373, -5.7457619, -8.9571953, -5.7410479, -2.4210677, 2.4190965
6: -19.4427872, -15.5619993, -19.4446411, -15.5569839, -3.1816673, 3.1770735
7: 4.2643223, 6.9667130, 4.2619171, 6.9752622, -2.7109399, 2.7047958
8: -7.1617842, -4.4029832, -7.1654897, -4.4018068, -2.3793960, 2.3808310
9: -7.2016177, -3.7783475, -7.2060957, -3.7777143, -2.6713328, 2.6756620

Time for backsubstitution: 21.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1791576, upper bound: 1.1791555
time: 4.55 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1791554, upper bound: 1.1791547
time: 4.63 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -17.6044273, -13.5805111, -17.5972481, -13.5857992, -2.5366182, 2.5315771
1: -10.2822809, -7.4614162, -10.2654266, -7.4666910, -2.2746010, 2.2608061
2: -6.4601746, -3.5581911, -6.4559097, -3.5972750, -2.3724961, 2.3958046
3: -2.4422810, 0.1332535, -2.4377654, 0.1256831, -1.8518305, 1.8501282
4: -7.0440598, -2.8905506, -6.9938722, -2.8966660, -3.1867456, 3.1572790
5: -8.9876623, -5.7355223, -8.9602032, -5.7369003, -2.4638071, 2.4334931
6: -19.4601669, -15.5480824, -19.4462547, -15.5525627, -3.2123308, 3.1964045
7: 4.2270660, 6.9874487, 4.2598314, 6.9827952, -2.7557292, 2.7276173
8: -7.1751170, -4.3977690, -7.1687737, -4.4007764, -2.3974919, 2.3905525
9: -7.2168636, -3.7630317, -7.2100434, -3.7771645, -2.6893601, 2.7021294

Time for backsubstitution: 21.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 478

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847108, upper bound: 1.1825616
time: 5.62 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847089, upper bound: 1.1847072
time: 4.50 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.48 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 31.48
Output dim: 7, lower bound: -1.1791576, upper bound: 1.1791555
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 31.48
Output dim: 7, lower bound: -1.1791554, upper bound: 1.1791547
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.48
Output dim: 7, lower bound: -1.1847108, upper bound: 1.1825616
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.48
Output dim: 7, lower bound: -1.1847089, upper bound: 1.1847072

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -17.5992355, -13.5838451, -17.5848007, -13.5906773, -2.5264196, 2.5153298
1: -10.2801342, -7.4985785, -10.2407980, -7.5379019, -2.2014403, 2.1992207
2: -6.4147644, -3.5602384, -6.3695850, -3.6265090, -2.2906842, 2.3072231
3: -2.4262748, 0.1311449, -2.4066973, 0.1139761, -1.8238640, 1.8172221
4: -7.0415030, -2.9192050, -6.9735928, -2.9521728, -3.1285367, 3.1081500
5: -8.9848471, -5.7430944, -8.9478683, -5.7516842, -2.4460173, 2.4110456
6: -19.4459839, -15.5498953, -19.4181862, -15.5658493, -3.1758595, 3.1672316
7: 4.2337799, 6.9838924, 4.2735138, 6.9688654, -2.7350855, 2.7103786
8: -7.1703587, -4.4277191, -7.1420603, -4.4574776, -2.3355908, 2.3215568
9: -7.2119217, -3.7737770, -7.1884127, -3.7976513, -2.6625123, 2.6593037

Time for backsubstitution: 21.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1809637, upper bound: 1.1748090
time: 5.07 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847067, upper bound: 1.1825585
time: 5.34 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -17.6044235, -13.5805092, -17.5972443, -13.5857983, -2.5366158, 2.5303392
1: -10.2822790, -7.4614162, -10.2654266, -7.4666939, -2.2241421, 2.2608032
2: -6.4601746, -3.5581908, -6.4559069, -3.5972748, -2.3656549, 2.3116672
3: -2.4422798, 0.1332532, -2.4377623, 0.1256831, -1.8518295, 1.8318923
4: -7.0440617, -2.8905525, -6.9938726, -2.8966670, -3.1436691, 3.1572781
5: -8.9876623, -5.7355223, -8.9602041, -5.7369032, -2.4513283, 2.4334903
6: -19.4601650, -15.5480824, -19.4462452, -15.5525627, -3.2092628, 3.1842585
7: 4.2270679, 6.9874487, 4.2598343, 6.9827919, -2.7557240, 2.7276144
8: -7.1751170, -4.3977714, -7.1687713, -4.4007769, -2.3481836, 2.3870080
9: -7.2168651, -3.7630305, -7.2100449, -3.7771668, -2.6919289, 2.6963220

Time for backsubstitution: 21.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1809637, upper bound: 1.1769540
time: 7.03 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847067, upper bound: 1.1847045
time: 4.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 33.38 seconds
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 33.38
Output dim: 7, lower bound: -1.1809637, upper bound: 1.1748090
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 33.38
Output dim: 7, lower bound: -1.1847067, upper bound: 1.1825585
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 33.38
Output dim: 7, lower bound: -1.1809637, upper bound: 1.1769540
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.38
Output dim: 7, lower bound: -1.1847067, upper bound: 1.1847045

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -17.5992355, -13.5838480, -17.5848007, -13.5906773, -2.5256362, 2.5055854
1: -10.2801313, -7.4985781, -10.2407980, -7.5379019, -2.2014399, 2.2017989
2: -6.4147615, -3.5602384, -6.3695850, -3.6265090, -2.2766199, 2.3042376
3: -2.4262705, 0.1311437, -2.4066973, 0.1139761, -1.7813673, 1.8172216
4: -7.0415039, -2.9192057, -6.9735928, -2.9521728, -3.1268554, 3.1004972
5: -8.9848433, -5.7430935, -8.9478683, -5.7516842, -2.4019704, 2.4110460
6: -19.4459820, -15.5498943, -19.4181862, -15.5658493, -3.1758585, 3.1456199
7: 4.2337813, 6.9838877, 4.2735138, 6.9688654, -2.7350841, 2.7103739
8: -7.1703572, -4.4277201, -7.1420603, -4.4574776, -2.3352385, 2.3004236
9: -7.2119193, -3.7737780, -7.1884127, -3.7976513, -2.6625113, 2.6540194

Time for backsubstitution: 21.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1791521, upper bound: 1.1819982
time: 4.99 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791521, upper bound: 1.1825595
time: 4.56 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -17.6044273, -13.5805092, -17.5972443, -13.5857983, -2.5358310, 2.5205884
1: -10.2822781, -7.4614153, -10.2654266, -7.4666939, -2.2241411, 2.2633824
2: -6.4601698, -3.5581913, -6.4559069, -3.5972748, -2.3515892, 2.3086791
3: -2.4422755, 0.1332514, -2.4377623, 0.1256831, -1.8093333, 1.8318923
4: -7.0440607, -2.8905520, -6.9938726, -2.8966670, -3.1419868, 3.1496248
5: -8.9876595, -5.7355213, -8.9602041, -5.7369032, -2.4072814, 2.4334898
6: -19.4601650, -15.5480843, -19.4462452, -15.5525627, -3.2092628, 3.1626492
7: 4.2270679, 6.9874449, 4.2598343, 6.9827919, -2.7557240, 2.7276106
8: -7.1751156, -4.3977733, -7.1687713, -4.4007769, -2.3478312, 2.3658905
9: -7.2168612, -3.7630329, -7.2100449, -3.7771668, -2.6919270, 2.6910357

Time for backsubstitution: 21.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791521, upper bound: 1.1841446
time: 4.71 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791521, upper bound: 1.1847052
time: 5.25 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.56 seconds
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 31.56
Output dim: 7, lower bound: -1.1791521, upper bound: 1.1819982
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.56
Output dim: 7, lower bound: -1.1791521, upper bound: 1.1825595
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.56
Output dim: 7, lower bound: -1.1791521, upper bound: 1.1841446
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.56
Output dim: 7, lower bound: -1.1791521, upper bound: 1.1847052

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -17.5992355, -13.5838480, -17.5919857, -13.5853910, -2.5290422, 2.5139780
1: -10.2801313, -7.4985781, -10.2576609, -7.5326190, -2.1949582, 2.2091355
2: -6.4147615, -3.5602384, -6.3738647, -3.5874135, -2.2830806, 2.3075700
3: -2.4262705, 0.1311437, -2.4112444, 0.1215166, -1.7885547, 1.8261895
4: -7.0415039, -2.9192057, -7.0237703, -2.9460711, -3.1112137, 3.1126699
5: -8.9848433, -5.7430935, -8.9753380, -5.7503128, -2.4033804, 2.4418883
6: -19.4459820, -15.5498943, -19.4321251, -15.5613804, -3.1814079, 3.1671500
7: 4.2337813, 6.9838877, 4.2407608, 6.9734745, -2.7396932, 2.7431269
8: -7.1703572, -4.4277201, -7.1483374, -4.4544740, -2.3386455, 2.3107839
9: -7.2119193, -3.7737780, -7.1951990, -3.7835126, -2.6816387, 2.6602969

Time for backsubstitution: 21.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 6209

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1760478, upper bound: 1.1770333
time: 6.63 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791509, upper bound: 1.1825580
time: 4.65 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -17.6042671, -13.5809422, -17.5882168, -13.5900869, -2.5290360, 2.5114903
1: -10.2821770, -7.4614344, -10.2623739, -7.4767742, -2.2136750, 2.2596807
2: -6.4601302, -3.5582705, -6.4378510, -3.5996532, -2.3447824, 2.2903476
3: -2.4422736, 0.1331904, -2.4340096, 0.1182394, -1.7960219, 1.8274796
4: -7.0437393, -2.8905892, -6.9883199, -2.9186797, -3.1192136, 3.1443834
5: -8.9874878, -5.7355232, -8.9537373, -5.7457647, -2.3966932, 2.4271040
6: -19.4600716, -15.5481167, -19.4427834, -15.5619993, -3.1885338, 3.1575980
7: 4.2272768, 6.9874005, 4.2643232, 6.9667110, -2.7394342, 2.7230773
8: -7.1750169, -4.3978472, -7.1617842, -4.4029851, -2.3405027, 2.3571084
9: -7.2168145, -3.7635984, -7.2016163, -3.7783494, -2.6882715, 2.6730142

Time for backsubstitution: 22.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 6209

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1760455, upper bound: 1.1786110
time: 4.19 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791509, upper bound: 1.1841428
time: 4.55 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -17.6044273, -13.5805092, -17.6044273, -13.5805092, -2.5392466, 2.5289910
1: -10.2822781, -7.4614153, -10.2822809, -7.4614182, -2.2176700, 2.2699370
2: -6.4601698, -3.5581913, -6.4601712, -3.5581920, -2.3580670, 2.3129098
3: -2.4422755, 0.1332514, -2.4422777, 0.1332529, -1.8165450, 1.8408060
4: -7.0440607, -2.8905520, -7.0440569, -2.8905535, -3.1284065, 3.1617498
5: -8.9876595, -5.7355213, -8.9876633, -5.7355242, -2.4086990, 2.4619570
6: -19.4601650, -15.5480843, -19.4601593, -15.5480824, -3.2148209, 3.1841373
7: 4.2270679, 6.9874449, 4.2270679, 6.9874477, -2.7603798, 2.7603769
8: -7.1751156, -4.3977733, -7.1751161, -4.3977718, -2.3512559, 2.3762212
9: -7.2168612, -3.7630329, -7.2168622, -3.7630334, -2.7110586, 2.6974134

Time for backsubstitution: 22.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 6209

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1760455, upper bound: 1.1791787
time: 6.00 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791509, upper bound: 1.1847037
time: 5.21 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 34.02 seconds
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 34.02
Output dim: 7, lower bound: -1.1760478, upper bound: 1.1770333
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 34.02
Output dim: 7, lower bound: -1.1791509, upper bound: 1.1825580
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 34.02
Output dim: 7, lower bound: -1.1760455, upper bound: 1.1786110
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 34.02
Output dim: 7, lower bound: -1.1791509, upper bound: 1.1841428
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 34.02
Output dim: 7, lower bound: -1.1760455, upper bound: 1.1791787
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 34.02
Output dim: 7, lower bound: -1.1791509, upper bound: 1.1847037

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -17.5992393, -13.5838623, -17.5919876, -13.5853901, -2.5271053, 2.4647272
1: -10.2801294, -7.4985862, -10.2576637, -7.5326185, -2.1949563, 2.1827574
2: -6.4147592, -3.5602388, -6.3738642, -3.5874133, -2.2309623, 2.2989082
3: -2.4262688, 0.1311424, -2.4112444, 0.1215174, -1.7590899, 1.8230627
4: -7.0415025, -2.9192057, -7.0237703, -2.9460711, -3.1112108, 3.0955281
5: -8.9848404, -5.7430954, -8.9753370, -5.7503119, -2.3850927, 2.4370835
6: -19.4459820, -15.5499001, -19.4321251, -15.5613785, -3.1814060, 3.1382790
7: 4.2337828, 6.9838881, 4.2407608, 6.9734750, -2.7396922, 2.7431273
8: -7.1703558, -4.4277225, -7.1483374, -4.4544749, -2.3386436, 2.3078408
9: -7.2119179, -3.7737820, -7.1951995, -3.7835124, -2.6813879, 2.6379728

Time for backsubstitution: 23.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1724785, upper bound: 1.1788135
time: 5.24 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1724785, upper bound: 1.1825594
time: 5.06 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -17.6042614, -13.5809546, -17.5882168, -13.5900860, -2.5290356, 2.4622386
1: -10.2821751, -7.4614401, -10.2623730, -7.4767761, -2.2136750, 2.2331381
2: -6.4601278, -3.5582714, -6.4378529, -3.5996537, -2.2926497, 2.2807679
3: -2.4422696, 0.1331869, -2.4340086, 0.1182413, -1.7664919, 1.8254182
4: -7.0437384, -2.8905928, -6.9883199, -2.9186792, -3.1153712, 3.1269464
5: -8.9874840, -5.7355251, -8.9537392, -5.7457657, -2.3783722, 2.4271026
6: -19.4600677, -15.5481215, -19.4427814, -15.5620022, -3.1885338, 3.1287274
7: 4.2272792, 6.9874001, 4.2643242, 6.9667130, -2.7394338, 2.7230759
8: -7.1750150, -4.3978472, -7.1617842, -4.4029851, -2.3404989, 2.3541589
9: -7.2168131, -3.7636027, -7.2016163, -3.7783504, -2.6880174, 2.6506662

Time for backsubstitution: 22.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1713146, upper bound: 1.1803293
time: 4.77 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1713146, upper bound: 1.1841442
time: 5.01 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -17.6044235, -13.5805225, -17.6044273, -13.5805082, -2.5392456, 2.4797385
1: -10.2822790, -7.4614224, -10.2822800, -7.4614172, -2.2176681, 2.2433944
2: -6.4601698, -3.5581932, -6.4601727, -3.5581913, -2.3059340, 2.3033321
3: -2.4422712, 0.1332488, -2.4422779, 0.1332508, -1.7870150, 1.8367662
4: -7.0440578, -2.8905537, -7.0440607, -2.8905525, -3.1284056, 3.1443849
5: -8.9876537, -5.7355242, -8.9876633, -5.7355232, -2.3903766, 2.4571531
6: -19.4601631, -15.5480919, -19.4601650, -15.5480843, -3.2148170, 3.1552677
7: 4.2270679, 6.9874430, 4.2270679, 6.9874482, -2.7603803, 2.7603750
8: -7.1751146, -4.3977747, -7.1751146, -4.3977723, -2.3512530, 2.3732719
9: -7.2168617, -3.7630377, -7.2168636, -3.7630334, -2.7108054, 2.6750636

Time for backsubstitution: 22.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1724785, upper bound: 1.1809598
time: 5.00 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1724785, upper bound: 1.1847045
time: 5.37 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 33.16 seconds
NS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 33.16
Output dim: 7, lower bound: -1.1724785, upper bound: 1.1788135
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 33.16
Output dim: 7, lower bound: -1.1724785, upper bound: 1.1825594
NS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 33.16
Output dim: 7, lower bound: -1.1713146, upper bound: 1.1803293
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 33.16
Output dim: 7, lower bound: -1.1713146, upper bound: 1.1841442
NS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 33.16
Output dim: 7, lower bound: -1.1724785, upper bound: 1.1809598
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 33.16
Output dim: 7, lower bound: -1.1724785, upper bound: 1.1847045

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -17.5992393, -13.5838623, -17.5919857, -13.5853910, -2.5200305, 2.4647262
1: -10.2801294, -7.4985862, -10.2576609, -7.5326190, -2.1975355, 2.1827555
2: -6.4147592, -3.5602388, -6.3738618, -3.5874131, -2.2294755, 2.2935114
3: -2.4262688, 0.1311424, -2.4112406, 0.1215167, -1.7588401, 1.7836926
4: -7.0415025, -2.9192057, -7.0237708, -2.9460754, -3.1035566, 3.0954781
5: -8.9848404, -5.7430954, -8.9753323, -5.7503138, -2.3850942, 2.3987322
6: -19.4459820, -15.5499001, -19.4321270, -15.5613842, -3.1597958, 3.1382780
7: 4.2337828, 6.9838881, 4.2407603, 6.9734697, -2.7396870, 2.7431278
8: -7.1703558, -4.4277225, -7.1483364, -4.4544764, -2.3178272, 2.3078403
9: -7.2119179, -3.7737820, -7.1951990, -3.7835126, -2.6761026, 2.6379719

Time for backsubstitution: 22.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1705442, upper bound: 1.1788145
time: 7.05 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1705442, upper bound: 1.1825586
time: 4.68 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -17.6042614, -13.5809546, -17.5882149, -13.5900860, -2.5200181, 2.4622374
1: -10.2821751, -7.4614401, -10.2623730, -7.4767752, -2.2162538, 2.2331376
2: -6.4601278, -3.5582714, -6.4378476, -3.5996552, -2.2911615, 2.2748218
3: -2.4422696, 0.1331869, -2.4340050, 0.1182402, -1.7664924, 1.7849813
4: -7.0437384, -2.8905928, -6.9883204, -2.9186831, -3.1144409, 3.1260843
5: -8.9874840, -5.7355251, -8.9537344, -5.7457647, -2.3783727, 2.3830552
6: -19.4600677, -15.5481215, -19.4427776, -15.5620041, -3.1669226, 3.1287274
7: 4.2272792, 6.9874001, 4.2643261, 6.9667082, -2.7394290, 2.7230740
8: -7.1750150, -4.3978472, -7.1617832, -4.4029884, -2.3196936, 2.3541574
9: -7.2168131, -3.7636027, -7.2016149, -3.7783506, -2.6827340, 2.6506653

Time for backsubstitution: 22.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1691695, upper bound: 1.1841439
time: 5.00 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1691695, upper bound: 1.1819980
time: 5.14 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -17.6044235, -13.5805225, -17.6044235, -13.5805120, -2.5302219, 2.4797373
1: -10.2822790, -7.4614224, -10.2822800, -7.4614182, -2.2202482, 2.2433946
2: -6.4601698, -3.5581932, -6.4601688, -3.5581913, -2.3044434, 2.2994914
3: -2.4422712, 0.1332488, -2.4422731, 0.1332521, -1.7870140, 1.7983077
4: -7.0440578, -2.8905537, -7.0440598, -2.8905561, -3.1207523, 3.1435180
5: -8.9876537, -5.7355242, -8.9876595, -5.7355223, -2.3903751, 2.4211707
6: -19.4601631, -15.5480919, -19.4601631, -15.5480843, -3.1932077, 3.1552668
7: 4.2270679, 6.9874430, 4.2270684, 6.9874430, -2.7603750, 2.7603745
8: -7.1751146, -4.3977747, -7.1751156, -4.3977737, -2.3304477, 2.3732708
9: -7.2168617, -3.7630377, -7.2168632, -3.7630346, -2.7055202, 2.6750636

Time for backsubstitution: 22.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1703329, upper bound: 1.1847042
time: 4.94 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1703329, upper bound: 1.1825585
time: 4.49 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 32.17 seconds
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 32.17
Output dim: 7, lower bound: -1.1705442, upper bound: 1.1788145
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 32.17
Output dim: 7, lower bound: -1.1705442, upper bound: 1.1825586
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 32.17
Output dim: 7, lower bound: -1.1691695, upper bound: 1.1841439
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 32.17
Output dim: 7, lower bound: -1.1691695, upper bound: 1.1819980
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 32.17
Output dim: 7, lower bound: -1.1703329, upper bound: 1.1847042
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 32.17
Output dim: 7, lower bound: -1.1703329, upper bound: 1.1825585

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -17.6044235, -13.5805225, -17.5919857, -13.5853910, -2.5230832, 2.4683509
1: -10.2822781, -7.4614244, -10.2576609, -7.5326190, -2.1959214, 2.1877952
2: -6.4601665, -3.5581923, -6.3738618, -3.5874131, -2.2323575, 2.2763348
3: -2.4422696, 0.1332510, -2.4112406, 0.1215167, -1.7623167, 1.7860458
4: -7.0440555, -2.8905604, -7.0237708, -2.9460754, -3.1060152, 3.0992947
5: -8.9876537, -5.7355223, -8.9753323, -5.7503138, -2.3878207, 2.4080272
6: -19.4601574, -15.5480938, -19.4321270, -15.5613842, -3.1792831, 3.1375713
7: 4.2270699, 6.9874430, 4.2407603, 6.9734697, -2.7463999, 2.7466826
8: -7.1751127, -4.3977757, -7.1483364, -4.4544764, -2.3198190, 2.3204203
9: -7.2168617, -3.7630398, -7.1951990, -3.7835126, -2.6771989, 2.6523504

Time for backsubstitution: 22.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6209

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1694046, upper bound: 1.1794642
time: 5.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1694046, upper bound: 1.1825591
time: 5.06 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -17.5918236, -13.5858345, -17.5882149, -13.5900898, -2.5073986, 2.4587646
1: -10.2575569, -7.5326433, -10.2623720, -7.4767747, -2.1958938, 2.1619458
2: -6.3738189, -3.5874920, -6.4378476, -3.5996542, -2.2045121, 2.2608259
3: -2.4112353, 0.1214514, -2.4340026, 0.1182392, -1.7361135, 1.7793047
4: -7.0234480, -2.9461098, -6.9883165, -2.9186831, -3.0960407, 3.0703468
5: -8.9751577, -5.7503147, -8.9537306, -5.7457647, -2.3727846, 2.3680253
6: -19.4320297, -15.5614204, -19.4427795, -15.5620060, -3.1401558, 3.1238794
7: 4.2409697, 6.9734287, 4.2643237, 6.9667072, -2.7257376, 2.7091050
8: -7.1482363, -4.4545512, -7.1617813, -4.4029880, -2.3215675, 2.2977629
9: -7.1951513, -3.7840834, -7.2016134, -3.7783525, -2.6516647, 2.6307554

Time for backsubstitution: 23.34 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.70 + 555.52 = 613.21 seconds
