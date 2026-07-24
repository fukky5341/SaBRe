## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 1)
Time budget: 1800 seconds
Split limit: 100
Threshold: 17.8846163811


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444)
1: (-15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248)
2: (-12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801)
3: (-8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943)
4: (-12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427)
5: (-9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509)
6: (-27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0119476, 19.0119514)
7: (-13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115)
8: (-16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7938156, 31.7938194)
9: (-12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.4043579, 21.4043617)
10: (-13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6689148, 34.6689186)
11: (-22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8888054, 33.8888054)
12: (-20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1794128, 36.1794128)
13: (-21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8892784, 25.8892822)
14: (-43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3938179, 34.3938103)
15: (-15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4708977, 24.4708939)
16: (-21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3633194, 33.3633156)
17: (-33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4835052, 52.4835129)
18: (-17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4217415, 24.4217415)
19: (-20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5478134, 21.5478134)
20: (-10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7429657, 19.7429657)
21: (-20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665)
22: (-22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3948288, 31.3948288)
23: (-19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3794556, 22.3794518)
24: (-26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5211906, 21.5211945)
25: (-13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3763962, 21.3764000)
26: (-28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7256012, 37.7256012)
27: (-28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5362625, 24.5362701)
28: (-18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0137711, 24.0137749)
29: (-32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8249359, 35.8249359)
30: (-18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7941742, 25.7941742)
31: (-18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1398239, 25.1398239)
32: (-21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3143997, 22.3144035)
33: (-39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2729187, 32.2729187)
34: (-30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8428612, 27.8428650)
35: (-30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2664719, 26.2664680)
36: (-31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5797386, 24.5797386)
37: (-47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6503754, 32.6503754)
38: (-40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7989540, 27.7989521)
39: (-50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4803543, 34.4803543)
40: (-41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8075867, 31.8075829)
41: (-31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0617027, 20.0617027)
42: (-18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5602455, 19.5602474)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.84 + 29.72 = 32.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 10, lower bound: -17.9025189, upper bound: 17.9025189

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1314
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1313
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1651

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8992941, upper bound: 17.8707439
time: 17.78 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8992941, upper bound: 17.8992939
time: 18.52 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 36.45 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 36.45
Output dim: 10, lower bound: -17.8992941, upper bound: 17.8707439
IS_A2, status: Status.UNKNOWN, split count: 1, time: 36.45
Output dim: 10, lower bound: -17.8992941, upper bound: 17.8992939

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -28.6255093, 5.7930856, -28.6272793, 5.8094230, -34.4349327, 34.4203644
1: -15.2251434, 11.1352701, -15.2259121, 11.1437283, -26.3688717, 26.3611832
2: -12.3342991, 10.9022179, -12.3349762, 10.9160328, -23.2503319, 23.2371941
3: -8.9930210, 15.7852020, -8.9941616, 15.8011093, -24.7941303, 24.7793636
4: -12.7409334, 13.2668657, -12.7419138, 13.2774687, -26.0184021, 26.0087795
5: -9.9337006, 18.0416641, -9.9348230, 18.0559616, -27.9896622, 27.9764862
6: -27.4536705, -2.9562340, -27.4665203, -2.9549379, -18.9843788, 18.9979382
7: -13.2533560, 17.7111301, -13.2545185, 17.7187634, -30.9721184, 30.9656487
8: -16.9920235, 15.8024940, -16.9933662, 15.8154793, -31.7810974, 31.7687683
9: -12.2550011, 13.5712986, -12.2562017, 13.5872078, -21.3893280, 21.3744736
10: -13.0742912, 24.7216492, -13.0764370, 24.7453136, -34.6457672, 34.6241455
11: -22.6848392, 12.8449469, -22.6960201, 12.8473492, -33.8666115, 33.8751602
12: -20.8591709, 15.4395714, -20.8645515, 15.4437752, -36.1582718, 36.1650314
13: -21.0817719, 11.3025827, -21.0842648, 11.3199902, -25.8712540, 25.8555031
14: -43.0324860, 3.4210405, -43.0373383, 3.4416747, -34.3677711, 34.3521500
15: -15.1044292, 9.8577271, -15.1062946, 9.8758688, -24.4527893, 24.4360275
16: -21.1396294, 13.1514549, -21.1414185, 13.1591034, -33.3540802, 33.3488541
17: -33.8495483, 27.4984589, -33.8566284, 27.5126667, -52.4590912, 52.4518051
18: -17.6543846, 7.9889984, -17.6677837, 7.9907050, -24.3959465, 24.4070835
19: -20.0832596, 2.0506546, -20.0954781, 2.0512691, -21.5253372, 21.5368462
20: -10.1490479, 10.3012486, -10.1622200, 10.3021507, -19.7182617, 19.7307434
21: -20.6769218, 7.2297602, -20.6928787, 7.2312226, -27.9081440, 27.9226379
22: -22.9098930, 9.3754950, -22.9207954, 9.3765745, -31.3748245, 31.3840256
23: -19.3517685, 4.2955985, -19.3603783, 4.2967691, -22.3574638, 22.3677902
24: -26.7314415, -1.6693230, -26.7466755, -1.6683936, -21.4932213, 21.5072365
25: -13.2781763, 9.5205631, -13.2931147, 9.5214291, -21.3476410, 21.3623772
26: -28.9174175, 8.8135481, -28.9293404, 8.8146067, -37.7032089, 37.7140961
27: -28.5676422, 0.3525167, -28.5844345, 0.3537216, -24.5050125, 24.5207329
28: -18.5134926, 6.3459148, -18.5299606, 6.3470759, -23.9830322, 23.9984474
29: -32.0700150, 5.0893145, -32.0783501, 5.0908585, -35.8080063, 35.8149948
30: -18.4672394, 8.4194841, -18.4870148, 8.4212427, -25.7567902, 25.7753677
31: -17.9938583, 8.5273237, -18.0098000, 8.5286999, -25.1099815, 25.1245155
32: -21.4100571, 4.2306590, -21.4173241, 4.2326803, -22.2966499, 22.3034706
33: -39.3128166, 1.1133900, -39.3245392, 1.1148200, -32.2503204, 32.2607231
34: -30.8188400, 2.1944180, -30.8254986, 2.1956253, -27.8295135, 27.8349724
35: -30.3060951, 2.4776893, -30.3138962, 2.4784517, -26.2517014, 26.2586594
36: -31.7499657, 0.2059734, -31.7596302, 0.2065895, -24.5620308, 24.5706673
37: -47.3440094, -6.5138068, -47.3627243, -6.5122871, -32.6140060, 32.6322021
38: -40.6511612, -2.1652341, -40.6646576, -2.1640306, -27.7726364, 27.7853508
39: -50.5607529, -5.9465604, -50.5692596, -5.9458628, -34.4648132, 34.4718208
40: -41.7049942, -3.3854194, -41.7162857, -3.3844986, -31.7812195, 31.7943573
41: -31.1593208, -4.2238188, -31.1685181, -4.2223334, -20.0439987, 20.0521584
42: -18.1265030, 2.5865159, -18.1306648, 2.5879836, -19.5380478, 19.5479679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=105, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1313
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1708

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8982421, upper bound: 17.8598191
time: 27.27 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8982421, upper bound: 17.8696923
time: 27.22 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -28.6725674, 5.8214579, -28.6278000, 5.8201070, -34.4926758, 34.4492569
1: -15.2483616, 11.1516380, -15.2261744, 11.1492138, -26.3975754, 26.3778114
2: -12.3714943, 10.9271002, -12.3352833, 10.9253922, -23.2968864, 23.2623825
3: -9.0343380, 15.8122921, -8.9947300, 15.8114738, -24.8458118, 24.8070221
4: -12.7655144, 13.2862740, -12.7426014, 13.2842216, -26.0497360, 26.0288754
5: -9.9739227, 18.0668316, -9.9354591, 18.0656261, -28.0395489, 28.0022907
6: -27.4785366, -2.9274960, -27.4745636, -2.9541664, -19.0054646, 19.0417442
7: -13.2716341, 17.7262783, -13.2549019, 17.7237644, -30.9953995, 30.9811802
8: -17.0269775, 15.8250294, -16.9941483, 15.8238811, -31.8254089, 31.7909775
9: -12.2963686, 13.5996246, -12.2567177, 13.5979166, -21.4412155, 21.3987465
10: -13.1357470, 24.7664490, -13.0772200, 24.7615414, -34.7235565, 34.6636429
11: -22.7065201, 12.8714924, -22.7023983, 12.8486366, -33.8897896, 33.9093094
12: -20.8701324, 15.4551668, -20.8674545, 15.4466124, -36.1697311, 36.1974831
13: -21.1300430, 11.3363705, -21.0855942, 11.3315506, -25.9319916, 25.8863754
14: -43.0994301, 3.4585719, -43.0401535, 3.4556990, -34.4501534, 34.3827286
15: -15.1494083, 9.8889980, -15.1068630, 9.8872871, -24.5106468, 24.4673080
16: -21.1672459, 13.1667967, -21.1416130, 13.1639662, -33.3863754, 33.3619232
17: -33.9004974, 27.5224876, -33.8614197, 27.5216713, -52.5203705, 52.4754562
18: -17.6800613, 8.0239582, -17.6768456, 7.9914441, -24.4182053, 24.4521389
19: -20.1063519, 2.0845315, -20.1035309, 2.0514722, -21.5465240, 21.5792313
20: -10.1728945, 10.3402348, -10.1707821, 10.3025455, -19.7394066, 19.7785988
21: -20.7071857, 7.2748108, -20.7034149, 7.2320356, -27.9392204, 27.9782257
22: -22.9326115, 9.4038363, -22.9279137, 9.3765774, -31.3942795, 31.4194489
23: -19.3685951, 4.3201580, -19.3658752, 4.2974043, -22.3707733, 22.4074745
24: -26.7586708, -1.6229982, -26.7564526, -1.6681662, -21.5164833, 21.5633850
25: -13.3059673, 9.5624571, -13.3031311, 9.5213041, -21.3721848, 21.4173088
26: -28.9415245, 8.8420887, -28.9370842, 8.8149195, -37.7261581, 37.7513428
27: -28.5987911, 0.3990612, -28.5956573, 0.3543534, -24.5319443, 24.5794830
28: -18.5429420, 6.3883924, -18.5411949, 6.3475833, -24.0126038, 24.0528107
29: -32.0899277, 5.1121273, -32.0837135, 5.0916491, -35.8278732, 35.8446579
30: -18.5026798, 8.4698267, -18.4999657, 8.4222336, -25.7890396, 25.8392906
31: -18.0239925, 8.5692368, -18.0200691, 8.5291824, -25.1378860, 25.1765442
32: -21.4254761, 4.2503905, -21.4211063, 4.2340403, -22.3116188, 22.3357582
33: -39.3393173, 1.1508269, -39.3330574, 1.1151314, -32.2728882, 32.3087692
34: -30.8344193, 2.2143340, -30.8291740, 2.1960554, -27.8444443, 27.8572159
35: -30.3230209, 2.4986434, -30.3188019, 2.4786491, -26.2670593, 26.2844391
36: -31.7707634, 0.2305481, -31.7657280, 0.2069478, -24.5783310, 24.6013489
37: -47.3792114, -6.4696460, -47.3750763, -6.5115809, -32.6473541, 32.6898613
38: -40.6793137, -2.1402416, -40.6735992, -2.1634350, -27.7963562, 27.8214645
39: -50.5789795, -5.9272881, -50.5744934, -5.9457173, -34.4807358, 34.4975357
40: -41.7241364, -3.3578825, -41.7220535, -3.3842163, -31.8077774, 31.8420715
41: -31.1783237, -4.1956730, -31.1741905, -4.2213402, -20.0606728, 20.0856400
42: -18.1386204, 2.5993714, -18.1319618, 2.5887394, -19.5443535, 19.5895424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=105, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1314
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1708

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8982421, upper bound: 17.8883734
time: 54.31 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8982421, upper bound: 17.8982418
time: 16.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 73.52 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 73.52
Output dim: 10, lower bound: -17.8982421, upper bound: 17.8598191
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 73.52
Output dim: 10, lower bound: -17.8982421, upper bound: 17.8696923
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 73.52
Output dim: 10, lower bound: -17.8982421, upper bound: 17.8883734
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 73.52
Output dim: 10, lower bound: -17.8982421, upper bound: 17.8982418

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -28.6239948, 5.7831469, -28.6266174, 5.8051143, -34.4291077, 34.4097633
1: -15.2239408, 11.1237278, -15.2253876, 11.1387949, -26.3627357, 26.3491154
2: -12.3333683, 10.8953953, -12.3345776, 10.9131212, -23.2464905, 23.2299728
3: -8.9905357, 15.7830420, -8.9930887, 15.8001957, -24.7907314, 24.7761307
4: -12.7400179, 13.2641563, -12.7415314, 13.2762909, -26.0163078, 26.0056877
5: -9.9324627, 18.0397778, -9.9343023, 18.0551395, -27.9876022, 27.9740791
6: -27.4418259, -2.9566574, -27.4614334, -2.9551296, -18.9731350, 18.9928436
7: -13.2512789, 17.7024574, -13.2536106, 17.7150421, -30.9663200, 30.9560680
8: -16.9912319, 15.7958765, -16.9930115, 15.8126450, -31.7744598, 31.7611618
9: -12.2542095, 13.5665588, -12.2558699, 13.5851564, -21.3864059, 21.3688507
10: -13.0728264, 24.7058105, -13.0757837, 24.7384529, -34.6377182, 34.6085358
11: -22.6807823, 12.8332176, -22.6942825, 12.8423119, -33.8574409, 33.8616142
12: -20.8497868, 15.4378490, -20.8604374, 15.4429893, -36.1483727, 36.1592865
13: -21.0721016, 11.3008671, -21.0801086, 11.3192768, -25.8626976, 25.8501129
14: -43.0278702, 3.4035563, -43.0353661, 3.4341640, -34.3558121, 34.3338585
15: -15.1036015, 9.8490276, -15.1059361, 9.8721218, -24.4486580, 24.4289398
16: -21.1363182, 13.1389894, -21.1399765, 13.1537142, -33.3456039, 33.3350449
17: -33.8424835, 27.4880924, -33.8535843, 27.5081425, -52.4473648, 52.4379807
18: -17.6509781, 7.9807453, -17.6663017, 7.9870787, -24.3888588, 24.3971138
19: -20.0801582, 2.0490382, -20.0941315, 2.0505285, -21.5215149, 21.5334206
20: -10.1452942, 10.3001366, -10.1605968, 10.3016748, -19.7132416, 19.7274399
21: -20.6725349, 7.2266078, -20.6909752, 7.2298279, -27.9023628, 27.9175835
22: -22.9049797, 9.3738155, -22.9186935, 9.3758316, -31.3661118, 31.3786240
23: -19.3485374, 4.2898607, -19.3589668, 4.2942934, -22.3518906, 22.3610840
24: -26.7291012, -1.6719236, -26.7456360, -1.6695061, -21.4897156, 21.5036240
25: -13.2748804, 9.5173931, -13.2916727, 9.5200644, -21.3421211, 21.3572578
26: -28.9131336, 8.8122244, -28.9274826, 8.8140411, -37.6939697, 37.7083206
27: -28.5660801, 0.3492794, -28.5837421, 0.3523464, -24.5011482, 24.5157585
28: -18.5099525, 6.3448744, -18.5283928, 6.3466277, -23.9778366, 23.9940605
29: -32.0672607, 5.0860214, -32.0771866, 5.0894022, -35.8019333, 35.8096466
30: -18.4619255, 8.4181995, -18.4847012, 8.4206963, -25.7502174, 25.7713394
31: -17.9907036, 8.5245895, -18.0084305, 8.5275316, -25.1054077, 25.1196404
32: -21.3951283, 4.2288370, -21.4108925, 4.2318835, -22.2814636, 22.2955971
33: -39.2952271, 1.1119390, -39.3168602, 1.1141849, -32.2316971, 32.2514725
34: -30.8010235, 2.1930499, -30.8178310, 2.1950207, -27.8109741, 27.8256493
35: -30.2879829, 2.4768543, -30.3059654, 2.4780803, -26.2329254, 26.2497444
36: -31.7316799, 0.2050941, -31.7517586, 0.2061927, -24.5431366, 24.5618134
37: -47.3315048, -6.5148921, -47.3573494, -6.5127859, -32.5994110, 32.6249390
38: -40.6352463, -2.1668825, -40.6577072, -2.1647320, -27.7551575, 27.7764244
39: -50.5398178, -5.9478264, -50.5602570, -5.9463983, -34.4425812, 34.4611816
40: -41.6907043, -3.3859625, -41.7100449, -3.3847280, -31.7664070, 31.7876205
41: -31.1500244, -4.2249470, -31.1644268, -4.2228160, -20.0346870, 20.0473862
42: -18.1236229, 2.5852747, -18.1294174, 2.5874338, -19.5347176, 19.5455647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=104, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1313
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 658

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8839416, upper bound: 17.8585344
time: 17.80 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8839416, upper bound: 17.8595115
time: 20.48 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -28.6423569, 5.7951031, -28.6270905, 5.8085580, -34.4509163, 34.4221954
1: -15.2436218, 11.1373396, -15.2257977, 11.1429100, -26.3865318, 26.3631363
2: -12.3460455, 10.9040575, -12.3348894, 10.9155369, -23.2615814, 23.2389469
3: -8.9958401, 15.7889957, -8.9936047, 15.8009233, -24.7967644, 24.7826004
4: -12.7435942, 13.2685986, -12.7418308, 13.2770214, -26.0206146, 26.0104294
5: -9.9359665, 18.0424519, -9.9346905, 18.0555744, -27.9915409, 27.9771423
6: -27.4534912, -2.9400740, -27.4643860, -2.9550095, -18.9829960, 19.0111885
7: -13.2642155, 17.7118378, -13.2542000, 17.7178078, -30.9820232, 30.9660378
8: -17.0009632, 15.8046169, -16.9932499, 15.8147011, -31.7763290, 31.7744637
9: -12.2617836, 13.5716820, -12.2561455, 13.5859604, -21.3987579, 21.3751373
10: -13.1017675, 24.7226410, -13.0762854, 24.7438679, -34.6678238, 34.6219368
11: -22.7075806, 12.8455620, -22.6953220, 12.8461046, -33.8889542, 33.8748207
12: -20.8601952, 15.4462223, -20.8635712, 15.4434929, -36.1578827, 36.1694374
13: -21.0829086, 11.3200178, -21.0830002, 11.3197222, -25.8737144, 25.8624763
14: -43.0609894, 3.4217997, -43.0367889, 3.4404850, -34.3941345, 34.3470688
15: -15.1207829, 9.8568277, -15.1061764, 9.8738422, -24.4591446, 24.4405556
16: -21.1651764, 13.1517906, -21.1409798, 13.1579981, -33.3807144, 33.3488846
17: -33.8672829, 27.4979210, -33.8559685, 27.5109921, -52.4758301, 52.4501343
18: -17.6680298, 7.9910727, -17.6673813, 7.9895482, -24.4097862, 24.4083004
19: -20.0938702, 2.0512972, -20.0951843, 2.0507941, -21.5349884, 21.5376205
20: -10.1498222, 10.3054371, -10.1615257, 10.3020706, -19.7177277, 19.7356949
21: -20.6874752, 7.2304535, -20.6923828, 7.2306204, -27.9180946, 27.9228363
22: -22.9136791, 9.3783875, -22.9196968, 9.3762226, -31.3734131, 31.3923492
23: -19.3667526, 4.2960787, -19.3600044, 4.2957973, -22.3709145, 22.3677940
24: -26.7376347, -1.6675510, -26.7463531, -1.6693115, -21.4968567, 21.5105171
25: -13.2870779, 9.5214510, -13.2927761, 9.5206547, -21.3484955, 21.3655891
26: -28.9214134, 8.8170233, -28.9288063, 8.8144827, -37.6955032, 37.7291412
27: -28.5733242, 0.3535590, -28.5841541, 0.3524184, -24.5060883, 24.5226974
28: -18.5173607, 6.3481102, -18.5295086, 6.3469601, -23.9830933, 24.0012817
29: -32.0787430, 5.0883408, -32.0780945, 5.0895557, -35.8109741, 35.8150940
30: -18.4691200, 8.4225407, -18.4858894, 8.4211073, -25.7571869, 25.7774773
31: -18.0093861, 8.5278721, -18.0094814, 8.5279312, -25.1228218, 25.1256199
32: -21.4116020, 4.2536850, -21.4161472, 4.2324181, -22.2944374, 22.3252869
33: -39.3148422, 1.1446290, -39.3226013, 1.1145172, -32.2492828, 32.2914200
34: -30.8217430, 2.2262297, -30.8242245, 2.1954651, -27.8293686, 27.8651123
35: -30.3081741, 2.5051665, -30.3125610, 2.4782882, -26.2515678, 26.2849007
36: -31.7511520, 0.2357161, -31.7583179, 0.2065065, -24.5591087, 24.5990601
37: -47.3455811, -6.4959784, -47.3603210, -6.5125184, -32.6135101, 32.6511040
38: -40.6516151, -2.1407032, -40.6632423, -2.1642160, -27.7693062, 27.8081207
39: -50.5632553, -5.9092999, -50.5677414, -5.9461403, -34.4633636, 34.5081177
40: -41.7071991, -3.3632426, -41.7151489, -3.3845539, -31.7812042, 31.8159790
41: -31.1612186, -4.2109966, -31.1675644, -4.2225223, -20.0425262, 20.0629063
42: -18.1295547, 2.5891633, -18.1300030, 2.5878029, -19.5407066, 19.5503483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=104, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 658

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8839416, upper bound: 17.8684054
time: 38.75 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8839416, upper bound: 17.8693848
time: 21.75 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -28.6710339, 5.8115311, -28.6271229, 5.8158197, -34.4868546, 34.4386520
1: -15.2471561, 11.1401024, -15.2256413, 11.1442595, -26.3914146, 26.3657436
2: -12.3705597, 10.9203243, -12.3348837, 10.9224873, -23.2930470, 23.2552071
3: -9.0318642, 15.8101110, -8.9936647, 15.8105402, -24.8424034, 24.8037758
4: -12.7646189, 13.2835455, -12.7421913, 13.2830534, -26.0476723, 26.0257378
5: -9.9726810, 18.0649338, -9.9349279, 18.0648041, -28.0374851, 27.9998627
6: -27.4667377, -2.9278717, -27.4694767, -2.9543438, -18.9942093, 19.0366516
7: -13.2695465, 17.7176094, -13.2539835, 17.7200356, -30.9895821, 30.9715919
8: -17.0262089, 15.8184328, -16.9938087, 15.8209839, -31.8187637, 31.7833824
9: -12.2955780, 13.5948830, -12.2563877, 13.5958595, -21.4382706, 21.3931160
10: -13.1343002, 24.7505932, -13.0766068, 24.7546654, -34.7155533, 34.6480408
11: -22.7024841, 12.8597736, -22.7006493, 12.8436050, -33.8806458, 33.8957863
12: -20.8607254, 15.4534378, -20.8632889, 15.4458427, -36.1598053, 36.1917152
13: -21.1203880, 11.3346472, -21.0814304, 11.3307838, -25.9234200, 25.8809929
14: -43.0948257, 3.4411182, -43.0381927, 3.4482088, -34.4382019, 34.3643913
15: -15.1485910, 9.8802700, -15.1065159, 9.8835659, -24.5065613, 24.4602318
16: -21.1639366, 13.1543417, -21.1401482, 13.1585674, -33.3778381, 33.3480759
17: -33.8934250, 27.5121536, -33.8583794, 27.5171928, -52.5086365, 52.4616470
18: -17.6766472, 8.0156860, -17.6753483, 7.9878182, -24.4111252, 24.4421768
19: -20.1032372, 2.0828915, -20.1021805, 2.0507479, -21.5426865, 21.5757751
20: -10.1691341, 10.3391056, -10.1691551, 10.3020668, -19.7343674, 19.7753105
21: -20.7027950, 7.2716761, -20.7015228, 7.2306600, -27.9334545, 27.9731979
22: -22.9276962, 9.4021721, -22.9257946, 9.3758507, -31.3855896, 31.4140472
23: -19.3653622, 4.3144169, -19.3644485, 4.2949219, -22.3652115, 22.4007797
24: -26.7563133, -1.6255865, -26.7554131, -1.6692548, -21.5129700, 21.5597954
25: -13.3026485, 9.5592890, -13.3016901, 9.5199280, -21.3666687, 21.4121895
26: -28.9372597, 8.8407211, -28.9352264, 8.8143444, -37.7169342, 37.7455826
27: -28.5972042, 0.3958325, -28.5949593, 0.3529606, -24.5281029, 24.5745468
28: -18.5393829, 6.3873386, -18.5396500, 6.3471355, -24.0074158, 24.0484314
29: -32.0872116, 5.1089096, -32.0825272, 5.0901880, -35.8217850, 35.8392944
30: -18.4973545, 8.4685345, -18.4976387, 8.4217205, -25.7824860, 25.8352623
31: -18.0208511, 8.5664864, -18.0187035, 8.5280218, -25.1332970, 25.1716614
32: -21.4105644, 4.2485962, -21.4146919, 4.2332425, -22.2964325, 22.3278580
33: -39.3217773, 1.1493621, -39.3253403, 1.1144905, -32.2542953, 32.2994461
34: -30.8165684, 2.2129588, -30.8215103, 2.1954360, -27.8258896, 27.8479080
35: -30.3049183, 2.4978466, -30.3109169, 2.4783230, -26.2482872, 26.2755394
36: -31.7524586, 0.2296848, -31.7578735, 0.2066045, -24.5594788, 24.5924721
37: -47.3667221, -6.4706864, -47.3696861, -6.5120783, -32.6327209, 32.6825638
38: -40.6633568, -2.1419106, -40.6666641, -2.1641626, -27.7788811, 27.8125534
39: -50.5580521, -5.9285450, -50.5654831, -5.9462643, -34.4585266, 34.4868889
40: -41.7098312, -3.3584261, -41.7157822, -3.3844514, -31.7929611, 31.8353195
41: -31.1690483, -4.1968288, -31.1700630, -4.2218299, -20.0513515, 20.0808678
42: -18.1357346, 2.5981145, -18.1307106, 2.5881929, -19.5410042, 19.5871277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=104, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 658

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8839416, upper bound: 17.8870684
time: 16.84 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8979339, upper bound: 17.8880653
time: 18.01 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -28.6893997, 5.8234782, -28.6276207, 5.8192415, -34.5086403, 34.4510994
1: -15.2668476, 11.1537037, -15.2260637, 11.1483736, -26.4152222, 26.3797684
2: -12.3832436, 10.9289703, -12.3351860, 10.9248867, -23.3081303, 23.2641563
3: -9.0371437, 15.8160553, -8.9941711, 15.8112621, -24.8484058, 24.8102264
4: -12.7681904, 13.2880116, -12.7425098, 13.2837715, -26.0519619, 26.0305214
5: -9.9761934, 18.0676003, -9.9353390, 18.0652618, -28.0414543, 28.0029392
6: -27.4783611, -2.9113092, -27.4724522, -2.9542494, -19.0040798, 19.0550022
7: -13.2824821, 17.7270012, -13.2545872, 17.7228470, -31.0053291, 30.9815884
8: -17.0359306, 15.8271847, -16.9940300, 15.8230648, -31.8206635, 31.7966843
9: -12.3031721, 13.6000080, -12.2566376, 13.5966778, -21.4506378, 21.3993950
10: -13.1632357, 24.7674065, -13.0770760, 24.7600822, -34.7456512, 34.6613998
11: -22.7292557, 12.8720913, -22.7017136, 12.8473997, -33.9121857, 33.9089622
12: -20.8711510, 15.4618206, -20.8664513, 15.4463434, -36.1693306, 36.2018738
13: -21.1311836, 11.3538380, -21.0843277, 11.3312759, -25.9344635, 25.8933563
14: -43.1279602, 3.4593191, -43.0396500, 3.4545317, -34.4765015, 34.3776054
15: -15.1657429, 9.8880711, -15.1067505, 9.8852901, -24.5170212, 24.4718399
16: -21.1927834, 13.1671391, -21.1411762, 13.1628857, -33.4130020, 33.3619385
17: -33.9182129, 27.5219803, -33.8607979, 27.5200310, -52.5371094, 52.4738159
18: -17.6936913, 8.0260277, -17.6764202, 7.9902887, -24.4320412, 24.4533539
19: -20.1169586, 2.0851450, -20.1032333, 2.0510011, -21.5561943, 21.5799942
20: -10.1736765, 10.3444023, -10.1700850, 10.3024368, -19.7388344, 19.7835693
21: -20.7177639, 7.2755232, -20.7029171, 7.2314339, -27.9491978, 27.9784393
22: -22.9364071, 9.4067268, -22.9268150, 9.3762245, -31.3928604, 31.4277725
23: -19.3835773, 4.3206472, -19.3655281, 4.2964401, -22.3842316, 22.4075012
24: -26.7648315, -1.6212225, -26.7561417, -1.6690683, -21.5201454, 21.5667038
25: -13.3148289, 9.5633364, -13.3027754, 9.5205135, -21.3730392, 21.4205284
26: -28.9455223, 8.8455324, -28.9365463, 8.8147573, -37.7184601, 37.7664261
27: -28.6045017, 0.4001102, -28.5953770, 0.3530698, -24.5330200, 24.5814972
28: -18.5467949, 6.3905945, -18.5407524, 6.3474693, -24.0126648, 24.0556755
29: -32.0987129, 5.1112127, -32.0834618, 5.0903234, -35.8308716, 35.8447342
30: -18.5045509, 8.4728928, -18.4988461, 8.4221058, -25.7894402, 25.8414078
31: -18.0395470, 8.5697441, -18.0197639, 8.5283928, -25.1507149, 25.1776199
32: -21.4270172, 4.2734275, -21.4199314, 4.2337761, -22.3093681, 22.3576012
33: -39.3413162, 1.1820827, -39.3311386, 1.1148210, -32.2718811, 32.3394318
34: -30.8373394, 2.2461329, -30.8278999, 2.1959176, -27.8442764, 27.8873367
35: -30.3250923, 2.5261154, -30.3174763, 2.4785357, -26.2668915, 26.3106842
36: -31.7719307, 0.2603180, -31.7644634, 0.2068894, -24.5754204, 24.6296997
37: -47.3808136, -6.4517293, -47.3726730, -6.5117683, -32.6468468, 32.7087631
38: -40.6797371, -2.1157198, -40.6722031, -2.1635919, -27.7930222, 27.8442326
39: -50.5814743, -5.8900361, -50.5729523, -5.9459782, -34.4792633, 34.5338593
40: -41.7263260, -3.3357124, -41.7208977, -3.3842726, -31.8077545, 31.8636856
41: -31.1802177, -4.1828718, -31.1731987, -4.2215056, -20.0591965, 20.0963974
42: -18.1416721, 2.6020126, -18.1312904, 2.5885582, -19.5470009, 19.5919113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=104, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1314
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 658

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8839416, upper bound: 17.8969338
time: 18.66 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8839416, upper bound: 17.8979339
time: 20.98 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 41.92 seconds
IS_A1_A1_A1, status: Status.VERIFIED, split count: 3, time: 41.92
Output dim: 10, lower bound: -17.8839416, upper bound: 17.8585344
IS_A1_A1_A2, status: Status.VERIFIED, split count: 3, time: 41.92
Output dim: 10, lower bound: -17.8839416, upper bound: 17.8595115
IS_A1_A2_A1, status: Status.VERIFIED, split count: 3, time: 41.92
Output dim: 10, lower bound: -17.8839416, upper bound: 17.8684054
IS_A1_A2_A2, status: Status.VERIFIED, split count: 3, time: 41.92
Output dim: 10, lower bound: -17.8839416, upper bound: 17.8693848
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 41.92
Output dim: 10, lower bound: -17.8839416, upper bound: 17.8870684
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 41.92
Output dim: 10, lower bound: -17.8979339, upper bound: 17.8880653
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 41.92
Output dim: 10, lower bound: -17.8839416, upper bound: 17.8969338
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 41.92
Output dim: 10, lower bound: -17.8839416, upper bound: 17.8979339

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: -28.6407280, 5.7870374, -28.6135769, 5.8144174, -34.4551468, 34.4006157
1: -15.2240963, 11.1166372, -15.2146807, 11.1426210, -26.3667183, 26.3313179
2: -12.3576841, 10.9063053, -12.3289871, 10.9208469, -23.2785301, 23.2352924
3: -9.0213947, 15.7995033, -8.9889784, 15.8089762, -24.8303719, 24.7884827
4: -12.7477589, 13.2652779, -12.7342014, 13.2816229, -26.0293808, 25.9994793
5: -9.9629612, 18.0554390, -9.9306984, 18.0640144, -28.0269756, 27.9861374
6: -27.4576645, -2.9340696, -27.4657116, -2.9557738, -18.9723473, 19.0287476
7: -13.2503834, 17.6995392, -13.2453480, 17.7186298, -30.9690132, 30.9448872
8: -17.0055523, 15.7942848, -16.9838734, 15.8184795, -31.7953186, 31.7486496
9: -12.2817144, 13.5860329, -12.2502670, 13.5949631, -21.4229698, 21.3758068
10: -13.1057167, 24.7271214, -13.0638580, 24.7527485, -34.6853790, 34.6122894
11: -22.6907578, 12.8539696, -22.6963329, 12.8417377, -33.8657150, 33.8831139
12: -20.8523064, 15.4423656, -20.8616028, 15.4438562, -36.1486053, 36.1784706
13: -21.1069260, 11.3198185, -21.0795937, 11.3243046, -25.9073944, 25.8654976
14: -43.0697823, 3.4223185, -43.0285110, 3.4462008, -34.4111862, 34.3361588
15: -15.1366520, 9.8714705, -15.1018963, 9.8804579, -24.4917870, 24.4460678
16: -21.1331272, 13.1343117, -21.1260986, 13.1572733, -33.3458710, 33.3141747
17: -33.8759346, 27.5006256, -33.8531532, 27.5155315, -52.4891205, 52.4407883
18: -17.6627922, 8.0040216, -17.6701431, 7.9854898, -24.3943100, 24.4247894
19: -20.0890484, 2.0697141, -20.0998039, 2.0448456, -21.5224190, 21.5599136
20: -10.1614590, 10.3330097, -10.1672125, 10.2996054, -19.7189255, 19.7670097
21: -20.6927948, 7.2687969, -20.6983643, 7.2293973, -27.9221916, 27.9671612
22: -22.9180603, 9.3952045, -22.9232578, 9.3729572, -31.3735733, 31.4048615
23: -19.3520050, 4.3004355, -19.3630219, 4.2884502, -22.3455811, 22.3850060
24: -26.7515526, -1.6283689, -26.7542896, -1.6711621, -21.5019188, 21.5507011
25: -13.2973585, 9.5529041, -13.3006411, 9.5171223, -21.3582153, 21.4046402
26: -28.9188194, 8.8239994, -28.9327679, 8.8069105, -37.6898346, 37.7262573
27: -28.5914383, 0.3915987, -28.5933228, 0.3516607, -24.5144272, 24.5683365
28: -18.5244637, 6.3719697, -18.5384598, 6.3397498, -23.9849548, 24.0316048
29: -32.0759163, 5.1063833, -32.0791817, 5.0894804, -35.8096390, 35.8338394
30: -18.4878788, 8.4649401, -18.4948711, 8.4195433, -25.7710419, 25.8294106
31: -18.0114632, 8.5561275, -18.0158157, 8.5239525, -25.1196365, 25.1581917
32: -21.4023857, 4.2434759, -21.4122505, 4.2317486, -22.2845764, 22.3203583
33: -39.3159180, 1.1405478, -39.3236580, 1.1106410, -32.2450027, 32.2891808
34: -30.8089695, 2.2028365, -30.8195953, 2.1910276, -27.8143616, 27.8358498
35: -30.2873383, 2.4811668, -30.3095379, 2.4704614, -26.2226181, 26.2572975
36: -31.7237854, 0.2034225, -31.7555275, 0.1939168, -24.5193787, 24.5645790
37: -47.3566284, -6.4762502, -47.3671417, -6.5144386, -32.6192474, 32.6744843
38: -40.6412239, -2.1571283, -40.6642303, -2.1708660, -27.7482109, 27.7937717
39: -50.5529366, -5.9365163, -50.5645828, -5.9497604, -34.4453812, 34.4767075
40: -41.6929245, -3.3688784, -41.7087250, -3.3865061, -31.7774773, 31.8267136
41: -31.1596107, -4.2059236, -31.1674881, -4.2256365, -20.0344810, 20.0687027
42: -18.1235695, 2.5897059, -18.1254463, 2.5869231, -19.5307655, 19.5818329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1748

## Relational analysis of IS_A2_A1_A1_A1

### Relational analysis result of IS_A2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8564569, upper bound: 17.8814546
time: 17.94 seconds

## Relational analysis of IS_A2_A1_A1_A2

### Relational analysis result of IS_A2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8564569, upper bound: 17.8814546
time: 20.38 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -28.6686020, 5.8113403, -28.6257286, 5.8157058, -34.4843063, 34.4370689
1: -15.2459393, 11.1398144, -15.2250328, 11.1440926, -26.3900318, 26.3648472
2: -12.3698549, 10.9200706, -12.3345222, 10.9223166, -23.2921715, 23.2545929
3: -9.0313969, 15.8094521, -8.9934244, 15.8101921, -24.8415890, 24.8028755
4: -12.7638760, 13.2831793, -12.7418079, 13.2828436, -26.0467186, 26.0249863
5: -9.9721670, 18.0648384, -9.9346695, 18.0647526, -28.0369186, 27.9995079
6: -27.4651299, -2.9280357, -27.4686680, -2.9544301, -19.0046825, 19.0288620
7: -13.2686558, 17.7173977, -13.2535477, 17.7198944, -30.9885502, 30.9709454
8: -17.0253334, 15.8175030, -16.9933510, 15.8204937, -31.8039932, 31.7820778
9: -12.2947617, 13.5945187, -12.2559299, 13.5956869, -21.4304047, 21.3924332
10: -13.1328669, 24.7503395, -13.0758753, 24.7545319, -34.7128067, 34.6471252
11: -22.7015972, 12.8585482, -22.7001629, 12.8429880, -33.8766785, 33.8962784
12: -20.8603401, 15.4531822, -20.8631058, 15.4457312, -36.1567459, 36.1908379
13: -21.1202049, 11.3334980, -21.0813522, 11.3301849, -25.9202194, 25.8748360
14: -43.0931625, 3.4407620, -43.0371971, 3.4479847, -34.4267464, 34.3632965
15: -15.1477032, 9.8779497, -15.1060581, 9.8823328, -24.5033188, 24.4563904
16: -21.1620674, 13.1542034, -21.1391678, 13.1585197, -33.3646088, 33.3469238
17: -33.8925171, 27.5116768, -33.8579140, 27.5169468, -52.5066071, 52.4621658
18: -17.6753311, 8.0154314, -17.6746254, 7.9876900, -24.4099197, 24.4414005
19: -20.1027775, 2.0823710, -20.1019440, 2.0504851, -21.5418930, 21.5726089
20: -10.1689129, 10.3385677, -10.1690416, 10.3017740, -19.7382545, 19.7727661
21: -20.7012329, 7.2715268, -20.7006569, 7.2305856, -27.9318180, 27.9721832
22: -22.9273071, 9.4018173, -22.9255867, 9.3756647, -31.3850174, 31.4129410
23: -19.3648262, 4.3137565, -19.3641739, 4.2945733, -22.3643036, 22.3932457
24: -26.7525616, -1.6258111, -26.7535515, -1.6693544, -21.5072174, 21.5568771
25: -13.3004293, 9.5589066, -13.3006048, 9.5197001, -21.3646469, 21.4107361
26: -28.9369545, 8.8396015, -28.9350643, 8.8137589, -37.7171249, 37.7438278
27: -28.5966759, 0.3955712, -28.5946999, 0.3528242, -24.5327682, 24.5715332
28: -18.5392456, 6.3864579, -18.5396061, 6.3466682, -24.0067520, 24.0458679
29: -32.0864067, 5.1088314, -32.0821152, 5.0901155, -35.8210297, 35.8383713
30: -18.4950142, 8.4682035, -18.4964409, 8.4215288, -25.7816734, 25.8331642
31: -18.0196590, 8.5661354, -18.0180550, 8.5278168, -25.1318245, 25.1702423
32: -21.4100151, 4.2481008, -21.4143867, 4.2329850, -22.2968254, 22.3269196
33: -39.3200645, 1.1489725, -39.3244705, 1.1142735, -32.2524109, 32.2980499
34: -30.8161201, 2.2108722, -30.8212757, 2.1943789, -27.8243637, 27.8430023
35: -30.3044682, 2.4970636, -30.3106537, 2.4779210, -26.2474518, 26.2650833
36: -31.7521820, 0.2285852, -31.7577591, 0.2060359, -24.5587387, 24.5810204
37: -47.3656998, -6.4710865, -47.3691216, -6.5122871, -32.6324463, 32.6811104
38: -40.6630516, -2.1433764, -40.6664734, -2.1650715, -27.7782860, 27.8005219
39: -50.5577087, -5.9291663, -50.5652657, -5.9466057, -34.4534378, 34.4810104
40: -41.7067299, -3.3589225, -41.7141190, -3.3846865, -31.7907639, 31.8313179
41: -31.1682720, -4.1990547, -31.1696701, -4.2229633, -20.0508919, 20.0795975
42: -18.1351700, 2.5978703, -18.1304169, 2.5880694, -19.5448761, 19.5839348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1748

## Relational analysis of IS_A2_A1_A2_A1

### Relational analysis result of IS_A2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8564569, upper bound: 17.8824946
time: 24.94 seconds

## Relational analysis of IS_A2_A1_A2_A2

### Relational analysis result of IS_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8971158, upper bound: 17.8872492
time: 20.86 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: -28.6590614, 5.7990060, -28.6140690, 5.8178558, -34.4769173, 34.4130745
1: -15.2437773, 11.1302414, -15.2151003, 11.1467247, -26.3905029, 26.3453407
2: -12.3703804, 10.9149446, -12.3292942, 10.9232559, -23.2936363, 23.2442398
3: -9.0267029, 15.8054714, -8.9894981, 15.8096809, -24.8363838, 24.7949696
4: -12.7513447, 13.2697716, -12.7345114, 13.2823296, -26.0336742, 26.0042839
5: -9.9664879, 18.0581398, -9.9310989, 18.0644817, -28.0309696, 27.9892387
6: -27.4693317, -2.9174914, -27.4686737, -2.9556599, -18.9822197, 19.0470924
7: -13.2633343, 17.7089272, -13.2459431, 17.7214165, -30.9847507, 30.9548702
8: -17.0152912, 15.8030491, -16.9841156, 15.8205299, -31.7971878, 31.7619743
9: -12.2892914, 13.5911484, -12.2505264, 13.5957661, -21.4353256, 21.3820934
10: -13.1346512, 24.7439537, -13.0643415, 24.7581654, -34.7154541, 34.6256065
11: -22.7175522, 12.8663139, -22.6973953, 12.8455391, -33.8972321, 33.8963203
12: -20.8627167, 15.4507408, -20.8647385, 15.4443436, -36.1581154, 36.1886330
13: -21.1177597, 11.3389750, -21.0824852, 11.3247528, -25.9184570, 25.8778534
14: -43.1029396, 3.4405642, -43.0299568, 3.4525204, -34.4494934, 34.3493462
15: -15.1538181, 9.8792763, -15.1021423, 9.8821907, -24.5022507, 24.4576836
16: -21.1619644, 13.1470966, -21.1271114, 13.1615639, -33.3810272, 33.3280296
17: -33.9007149, 27.5104523, -33.8555603, 27.5183697, -52.5176086, 52.4529724
18: -17.6798515, 8.0143642, -17.6712093, 7.9879799, -24.4152298, 24.4359684
19: -20.1027489, 2.0719633, -20.1008511, 2.0451043, -21.5359116, 21.5641327
20: -10.1659908, 10.3383179, -10.1681395, 10.3000078, -19.7234001, 19.7752914
21: -20.7077408, 7.2726402, -20.6997604, 7.2301683, -27.9379082, 27.9724007
22: -22.9267521, 9.3997288, -22.9242916, 9.3733387, -31.3808441, 31.4185715
23: -19.3702202, 4.3066463, -19.3640709, 4.2899590, -22.3646011, 22.3917427
24: -26.7600727, -1.6239734, -26.7550106, -1.6709871, -21.5090714, 21.5575867
25: -13.3095493, 9.5569658, -13.3017101, 9.5177183, -21.3646164, 21.4129944
26: -28.9270916, 8.8287830, -28.9341240, 8.8073578, -37.6913452, 37.7470932
27: -28.5987282, 0.3958755, -28.5937614, 0.3517399, -24.5193405, 24.5753059
28: -18.5318584, 6.3751936, -18.5395584, 6.3400965, -23.9902191, 24.0388794
29: -32.0873795, 5.1086702, -32.0801315, 5.0896339, -35.8187408, 35.8392715
30: -18.4950409, 8.4693155, -18.4960785, 8.4199543, -25.7779961, 25.8355370
31: -18.0301590, 8.5593948, -18.0168705, 8.5243444, -25.1370430, 25.1641560
32: -21.4188309, 4.2683215, -21.4174881, 4.2323060, -22.2975502, 22.3500710
33: -39.3355446, 1.1732354, -39.3294220, 1.1109653, -32.2625809, 32.3291397
34: -30.8296814, 2.2360163, -30.8260098, 2.1914716, -27.8327560, 27.8753052
35: -30.3075066, 2.5094700, -30.3161030, 2.4706607, -26.2412338, 26.2924461
36: -31.7432766, 0.2340517, -31.7621231, 0.1941762, -24.5353394, 24.6018066
37: -47.3706894, -6.4573145, -47.3701248, -6.5141292, -32.6333389, 32.7006912
38: -40.6576004, -2.1309505, -40.6697426, -2.1703496, -27.7623367, 27.8254604
39: -50.5763588, -5.8979487, -50.5720520, -5.9494734, -34.4661484, 34.5236664
40: -41.7094383, -3.3461676, -41.7138138, -3.3863125, -31.7922821, 31.8550682
41: -31.1708069, -4.1919827, -31.1706276, -4.2253127, -20.0423203, 20.0842285
42: -18.1294975, 2.5935998, -18.1260147, 2.5872941, -19.5367470, 19.5866146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1748

## Relational analysis of IS_A2_A2_A1_A1

### Relational analysis result of IS_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8564569, upper bound: 17.8913260
time: 21.69 seconds

## Relational analysis of IS_A2_A2_A1_A2

### Relational analysis result of IS_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8564569, upper bound: 17.8961102
time: 21.12 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: -28.6869659, 5.8232784, -28.6262360, 5.8191099, -34.5060768, 34.4495163
1: -15.2656240, 11.1534138, -15.2254286, 11.1482105, -26.4138336, 26.3788414
2: -12.3825340, 10.9287128, -12.3348083, 10.9247541, -23.3072891, 23.2635212
3: -9.0366802, 15.8153915, -8.9939299, 15.8109264, -24.8476067, 24.8093224
4: -12.7674503, 13.2876453, -12.7421398, 13.2835789, -26.0510292, 26.0297852
5: -9.9756870, 18.0674992, -9.9350958, 18.0652103, -28.0408974, 28.0025940
6: -27.4767895, -2.9114499, -27.4716225, -2.9543142, -19.0145416, 19.0472069
7: -13.2816010, 17.7267475, -13.2541590, 17.7227211, -31.0043221, 30.9809074
8: -17.0350723, 15.8262386, -16.9935856, 15.8225498, -31.8058929, 31.7953796
9: -12.3023577, 13.5996780, -12.2561893, 13.5964823, -21.4427719, 21.3987122
10: -13.1618099, 24.7671738, -13.0763340, 24.7599621, -34.7429504, 34.6604843
11: -22.7283916, 12.8708630, -22.7012329, 12.8467960, -33.9082031, 33.9095039
12: -20.8707466, 15.4615774, -20.8662720, 15.4462156, -36.1662674, 36.2010231
13: -21.1310158, 11.3526602, -21.0842514, 11.3306618, -25.9312515, 25.8871918
14: -43.1262512, 3.4589596, -43.0386353, 3.4543362, -34.4650764, 34.3764725
15: -15.1648865, 9.8857517, -15.1062956, 9.8840609, -24.5137939, 24.4679947
16: -21.1909103, 13.1669817, -21.1401711, 13.1627998, -33.3997498, 33.3608170
17: -33.9173203, 27.5214748, -33.8603210, 27.5197430, -52.5350800, 52.4743271
18: -17.6923809, 8.0257702, -17.6757164, 7.9901414, -24.4308434, 24.4525661
19: -20.1164932, 2.0846183, -20.1029758, 2.0507431, -21.5553970, 21.5768204
20: -10.1734581, 10.3438740, -10.1699600, 10.3021736, -19.7427292, 19.7810211
21: -20.7161961, 7.2753701, -20.7020721, 7.2313614, -27.9475574, 27.9774418
22: -22.9360256, 9.4063950, -22.9266319, 9.3760500, -31.3922958, 31.4266968
23: -19.3830357, 4.3199792, -19.3652229, 4.2960901, -22.3833313, 22.3999481
24: -26.7611008, -1.6214256, -26.7542572, -1.6691532, -21.5143814, 21.5637512
25: -13.3126354, 9.5629663, -13.3016663, 9.5202875, -21.3710289, 21.4190788
26: -28.9452724, 8.8444052, -28.9364014, 8.8141918, -37.7186279, 37.7646332
27: -28.6039677, 0.3998232, -28.5951138, 0.3529181, -24.5377045, 24.5784531
28: -18.5466557, 6.3896918, -18.5406780, 6.3470087, -24.0120239, 24.0530777
29: -32.0979004, 5.1110983, -32.0830269, 5.0902710, -35.8301086, 35.8438110
30: -18.5022278, 8.4725723, -18.4976482, 8.4219227, -25.7886276, 25.8392944
31: -18.0383492, 8.5693874, -18.0191193, 8.5282326, -25.1492310, 25.1762238
32: -21.4264565, 4.2729268, -21.4196510, 4.2335262, -22.3097916, 22.3566360
33: -39.3396492, 1.1816483, -39.3302269, 1.1145902, -32.2700043, 32.3380356
34: -30.8368320, 2.2440453, -30.8276482, 2.1948562, -27.8427353, 27.8824921
35: -30.3246174, 2.5253625, -30.3172264, 2.4781284, -26.2660561, 26.3002396
36: -31.7716618, 0.2592185, -31.7643356, 0.2063251, -24.5746460, 24.6182709
37: -47.3797684, -6.4521489, -47.3721199, -6.5119734, -32.6465416, 32.7073364
38: -40.6794281, -2.1172228, -40.6720505, -2.1645398, -27.7924309, 27.8322334
39: -50.5811615, -5.8906231, -50.5727386, -5.9462872, -34.4742432, 34.5279846
40: -41.7232780, -3.3362136, -41.7192154, -3.3845162, -31.8055496, 31.8596840
41: -31.1794872, -4.1851130, -31.1728001, -4.2226515, -20.0587311, 20.0951157
42: -18.1411018, 2.6017580, -18.1309853, 2.5884390, -19.5508614, 19.5887184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1314
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1748

## Relational analysis of IS_A2_A2_A2_A1

### Relational analysis result of IS_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8564569, upper bound: 17.8923682
time: 20.68 seconds

## Relational analysis of IS_A2_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8971158, upper bound: 17.8971161
time: 18.50 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 41.47 seconds
IS_A2_A1_A1_A1, status: Status.VERIFIED, split count: 4, time: 41.47
Output dim: 10, lower bound: -17.8564569, upper bound: 17.8814546
IS_A2_A1_A1_A2, status: Status.VERIFIED, split count: 4, time: 41.47
Output dim: 10, lower bound: -17.8564569, upper bound: 17.8814546
IS_A2_A1_A2_A1, status: Status.VERIFIED, split count: 4, time: 41.47
Output dim: 10, lower bound: -17.8564569, upper bound: 17.8824946
IS_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 41.47
Output dim: 10, lower bound: -17.8971158, upper bound: 17.8872492
IS_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 41.47
Output dim: 10, lower bound: -17.8564569, upper bound: 17.8913260
IS_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 41.47
Output dim: 10, lower bound: -17.8564569, upper bound: 17.8961102
IS_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 41.47
Output dim: 10, lower bound: -17.8564569, upper bound: 17.8923682
IS_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 41.47
Output dim: 10, lower bound: -17.8971158, upper bound: 17.8971161

## BFS IS instance: IS_A2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -28.6651955, 5.8110452, -28.6246109, 5.8156171, -34.4808121, 34.4356575
1: -15.2441759, 11.1393585, -15.2244539, 11.1439228, -26.3880997, 26.3638115
2: -12.3684511, 10.9197731, -12.3340569, 10.9222336, -23.2906837, 23.2538300
3: -9.0298271, 15.8087225, -8.9929113, 15.8099480, -24.8397751, 24.8016338
4: -12.7626667, 13.2827291, -12.7414207, 13.2827168, -26.0453835, 26.0241508
5: -9.9707003, 18.0645275, -9.9341927, 18.0646610, -28.0353622, 27.9987202
6: -27.4643421, -2.9314814, -27.4684029, -2.9555669, -19.0031834, 19.0138149
7: -13.2672482, 17.7168407, -13.2531013, 17.7197361, -30.9869843, 30.9699421
8: -17.0235348, 15.8164806, -16.9927559, 15.8201790, -31.8010406, 31.7860565
9: -12.2924700, 13.5940828, -12.2551575, 13.5955391, -21.3994179, 21.3911743
10: -13.1302729, 24.7498074, -13.0749950, 24.7543716, -34.6917953, 34.6456833
11: -22.7009144, 12.8563614, -22.6999493, 12.8422928, -33.8752480, 33.8860703
12: -20.8591423, 15.4525423, -20.8627377, 15.4455004, -36.1702919, 36.1864700
13: -21.1176548, 11.3327293, -21.0805130, 11.3299341, -25.8831635, 25.8730431
14: -43.0896530, 3.4402695, -43.0360107, 3.4478331, -34.3955917, 34.3618622
15: -15.1462107, 9.8773050, -15.1055584, 9.8821220, -24.5013161, 24.4571190
16: -21.1596413, 13.1537724, -21.1384029, 13.1583462, -33.3511963, 33.3456955
17: -33.8903770, 27.5108719, -33.8572502, 27.5166378, -52.5011826, 52.4572449
18: -17.6742020, 8.0133905, -17.6742516, 7.9870067, -24.4080124, 24.4123344
19: -20.1022873, 2.0805025, -20.1017723, 2.0498774, -21.5407982, 21.5554466
20: -10.1685047, 10.3365870, -10.1688976, 10.3011417, -19.7371902, 19.7630577
21: -20.7007370, 7.2692852, -20.7005043, 7.2298570, -27.9305935, 27.9697895
22: -22.9265823, 9.3990707, -22.9253731, 9.3747826, -31.3834000, 31.4073639
23: -19.3643322, 4.3119235, -19.3640289, 4.2939901, -22.3632355, 22.3823586
24: -26.7517014, -1.6283617, -26.7532597, -1.6701813, -21.5055466, 21.5256157
25: -13.3000212, 9.5568686, -13.3004522, 9.5190582, -21.3635674, 21.4063568
26: -28.9363880, 8.8366413, -28.9348793, 8.8127804, -37.7154694, 37.7230759
27: -28.5962067, 0.3927712, -28.5945396, 0.3519135, -24.5314102, 24.5370522
28: -18.5389824, 6.3843360, -18.5394821, 6.3459668, -24.0058212, 24.0345230
29: -32.0854149, 5.1056957, -32.0817719, 5.0891247, -35.8190079, 35.8317642
30: -18.4944649, 8.4658051, -18.4962387, 8.4207296, -25.7803078, 25.8235893
31: -18.0184975, 8.5642815, -18.0176907, 8.5272293, -25.1300049, 25.1547585
32: -21.4093437, 4.2467122, -21.4141579, 4.2325087, -22.3003998, 22.3248329
33: -39.3189278, 1.1467237, -39.3240662, 1.1135464, -32.2494659, 32.2936935
34: -30.8155670, 2.2092748, -30.8210716, 2.1938558, -27.8206863, 27.8459015
35: -30.3037281, 2.4952207, -30.3104038, 2.4773164, -26.2451096, 26.2621078
36: -31.7515507, 0.2259364, -31.7575302, 0.2051661, -24.5576744, 24.5686874
37: -47.3644371, -6.4741111, -47.3687248, -6.5132637, -32.6302261, 32.6554947
38: -40.6623383, -2.1452227, -40.6663094, -2.1656923, -27.7754517, 27.7856789
39: -50.5559998, -5.9308124, -50.5646896, -5.9471412, -34.4496078, 34.4808197
40: -41.7056351, -3.3608055, -41.7137070, -3.3853245, -31.7870941, 31.8214722
41: -31.1675301, -4.2024341, -31.1694298, -4.2240644, -20.0495644, 20.0550308
42: -18.1347752, 2.5966158, -18.1302814, 2.5876594, -19.5486183, 19.5817986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=102, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1313
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_A1_A2_A2_B1

### Relational analysis result of IS_A2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8879247, upper bound: 17.8419849
time: 28.97 seconds

## Relational analysis of IS_A2_A1_A2_A2_B2

### Relational analysis result of IS_A2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8961993, upper bound: 17.8863218
time: 26.10 seconds

## BFS IS instance: IS_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -28.5814667, 5.7887526, -28.5773697, 5.8149090, -34.3963776, 34.3661232
1: -15.2067089, 11.1205301, -15.1980238, 11.1434431, -26.3501511, 26.3185539
2: -12.3265047, 10.9045143, -12.3087616, 10.9202328, -23.2467384, 23.2132759
3: -8.9738750, 15.7818613, -8.9646769, 15.8025684, -24.7764435, 24.7465382
4: -12.7136383, 13.2525139, -12.7168665, 13.2767620, -25.9904003, 25.9693794
5: -9.9197254, 18.0422707, -9.9091969, 18.0604382, -27.9801636, 27.9514675
6: -27.4505215, -2.9635448, -27.4604549, -2.9773874, -18.9401550, 18.9916172
7: -13.2199993, 17.6974220, -13.2258883, 17.7162819, -30.9362812, 30.9233093
8: -16.9586563, 15.7844191, -16.9577160, 15.8134499, -31.7295990, 31.7061501
9: -12.2150583, 13.5621548, -12.2151966, 13.5894823, -21.3538895, 21.3165512
10: -13.0565233, 24.7123718, -13.0279789, 24.7495651, -34.6278152, 34.5575027
11: -22.6857719, 12.7953520, -22.6900826, 12.8118544, -33.8309097, 33.8167000
12: -20.8358803, 15.4330883, -20.8524380, 15.4372349, -36.1071053, 36.1462288
13: -21.0378647, 11.3060894, -21.0443954, 11.3173876, -25.8290596, 25.8054123
14: -43.0024338, 3.4249878, -42.9823532, 3.4480972, -34.3426208, 34.2855453
15: -15.1215868, 9.8610640, -15.0869303, 9.8746843, -24.4602432, 24.4203911
16: -21.0907440, 13.1300344, -21.0941734, 13.1570978, -33.3062363, 33.2783470
17: -33.8273811, 27.4931602, -33.8235626, 27.5101929, -52.4250717, 52.3955383
18: -17.6488914, 7.9473186, -17.6635742, 7.9560328, -24.3509560, 24.3594227
19: -20.0773888, 2.0098882, -20.0959206, 2.0153604, -21.4814606, 21.4977608
20: -10.1448336, 10.2739162, -10.1642914, 10.2692614, -19.6701965, 19.7047539
21: -20.6774330, 7.2000895, -20.6935368, 7.1955147, -27.8729477, 27.8936272
22: -22.8948116, 9.3079376, -22.9171391, 9.3293180, -31.3048706, 31.3194885
23: -19.3445549, 4.2506733, -19.3588943, 4.2632313, -22.3123245, 22.3297081
24: -26.7243881, -1.7084131, -26.7466850, -1.7114305, -21.4329910, 21.4650192
25: -13.2872810, 9.4936857, -13.2964411, 9.4873619, -21.3117599, 21.3430519
26: -28.8947620, 8.7332401, -28.9284744, 8.7617912, -37.6150513, 37.6472092
27: -28.5639420, 0.3049545, -28.5876541, 0.3083220, -24.4422836, 24.4788666
28: -18.5089912, 6.3096108, -18.5363941, 6.3087754, -23.9365082, 23.9702454
29: -32.0462494, 5.0016508, -32.0695763, 5.0384693, -35.7261581, 35.7208023
30: -18.4657593, 8.3972826, -18.4895973, 8.3856335, -25.7135658, 25.7555656
31: -17.9975281, 8.5008526, -18.0078011, 8.4963989, -25.0757217, 25.0957584
32: -21.3991699, 4.2381096, -21.4100380, 4.2183766, -22.2567291, 22.3056221
33: -39.3138542, 1.1403227, -39.3200760, 1.0958967, -32.2228012, 32.2826080
34: -30.8155098, 2.2034578, -30.8192329, 2.1767540, -27.8053284, 27.8355370
35: -30.2886581, 2.4849157, -30.3077431, 2.4596324, -26.2088814, 26.2555237
36: -31.7253036, 0.1802673, -31.7544708, 0.1684520, -24.4906464, 24.5364075
37: -47.3379593, -6.5177622, -47.3571930, -6.5425639, -32.5662155, 32.6186066
38: -40.6335831, -2.1881409, -40.6612167, -2.1968279, -27.7168999, 27.7567711
39: -50.5476227, -5.9176602, -50.5594101, -5.9579453, -34.4254608, 34.4868889
40: -41.6894379, -3.3793759, -41.7042656, -3.4014492, -31.7579269, 31.8109779
41: -31.1485043, -4.2456584, -31.1616249, -4.2507997, -19.9900894, 20.0138321
42: -18.1142540, 2.5577636, -18.1206818, 2.5704370, -19.4999695, 19.5432205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=102, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1314
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1313
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A2_A2_A1_A1_A1

### Relational analysis result of IS_A2_A2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8526792, upper bound: 17.8731976
time: 26.16 seconds

## Relational analysis of IS_A2_A2_A1_A1_A2

### Relational analysis result of IS_A2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8526792, upper bound: 17.8875334
time: 18.00 seconds

## BFS IS instance: IS_A2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -28.6556721, 5.7986865, -28.6129646, 5.8177485, -34.4734192, 34.4116516
1: -15.2420254, 11.1297550, -15.2145052, 11.1465759, -26.3886013, 26.3442612
2: -12.3689728, 10.9146557, -12.3288364, 10.9231586, -23.2921314, 23.2434921
3: -9.0251408, 15.8047495, -8.9889879, 15.8094540, -24.8345947, 24.7937374
4: -12.7501373, 13.2692890, -12.7341213, 13.2821951, -26.0323334, 26.0034103
5: -9.9650097, 18.0578537, -9.9306173, 18.0643749, -28.0293846, 27.9884720
6: -27.4685020, -2.9209294, -27.4684219, -2.9567976, -18.9807167, 19.0320339
7: -13.2619362, 17.7084198, -13.2454796, 17.7212334, -30.9831696, 30.9538994
8: -17.0134830, 15.8020077, -16.9835091, 15.8201666, -31.7942276, 31.7659607
9: -12.2869949, 13.5907011, -12.2497683, 13.5956097, -21.4043465, 21.3808517
10: -13.1320477, 24.7434120, -13.0635061, 24.7579727, -34.6944275, 34.6241989
11: -22.7168579, 12.8641481, -22.6971416, 12.8448143, -33.8957977, 33.8861008
12: -20.8615284, 15.4500980, -20.8643417, 15.4441242, -36.1716347, 36.1842499
13: -21.1151924, 11.3381729, -21.0816689, 11.3245287, -25.8813896, 25.8760567
14: -43.0994263, 3.4400949, -43.0288162, 3.4523568, -34.4183273, 34.3479118
15: -15.1523228, 9.8786316, -15.1016464, 9.8820000, -24.5002556, 24.4584122
16: -21.1595573, 13.1466532, -21.1263103, 13.1614056, -33.3675842, 33.3267975
17: -33.8985901, 27.5096188, -33.8548508, 27.5181007, -52.5121536, 52.4480209
18: -17.6787033, 8.0123081, -17.6708488, 7.9872994, -24.4133072, 24.4069099
19: -20.1022358, 2.0700946, -20.1006756, 2.0444989, -21.5348167, 21.5469742
20: -10.1655846, 10.3363400, -10.1679974, 10.2993622, -19.7223358, 19.7655830
21: -20.7072563, 7.2704010, -20.6995831, 7.2294707, -27.9367275, 27.9699841
22: -22.9260445, 9.3969707, -22.9240303, 9.3724346, -31.3792343, 31.4129791
23: -19.3697319, 4.3048077, -19.3638973, 4.2893877, -22.3635406, 22.3808632
24: -26.7592583, -1.6265397, -26.7547226, -1.6718016, -21.5073814, 21.5263176
25: -13.3091030, 9.5549288, -13.3015690, 9.5170584, -21.3635101, 21.4086380
26: -28.9265175, 8.8257856, -28.9339085, 8.8063660, -37.6896896, 37.7263565
27: -28.5982704, 0.3931055, -28.5936127, 0.3508348, -24.5179901, 24.5407867
28: -18.5316048, 6.3730564, -18.5394783, 6.3393917, -23.9892731, 24.0275345
29: -32.0863838, 5.1055632, -32.0797806, 5.0886192, -35.8167191, 35.8326874
30: -18.4944973, 8.4669123, -18.4958839, 8.4191742, -25.7765999, 25.8259506
31: -18.0289936, 8.5575466, -18.0164986, 8.5237551, -25.1352158, 25.1486664
32: -21.4181366, 4.2669172, -21.4172535, 4.2318301, -22.3011093, 22.3480072
33: -39.3344002, 1.1710110, -39.3290558, 1.1102552, -32.2596588, 32.3247910
34: -30.8291111, 2.2344322, -30.8257847, 2.1909518, -27.8290710, 27.8781853
35: -30.3067665, 2.5076399, -30.3158512, 2.4700551, -26.2388725, 26.2895126
36: -31.7426376, 0.2313888, -31.7619152, 0.1933234, -24.5342369, 24.5894585
37: -47.3694458, -6.4603734, -47.3697319, -6.5151291, -32.6310997, 32.6750107
38: -40.6568985, -2.1327791, -40.6695213, -2.1709533, -27.7595062, 27.8106251
39: -50.5746155, -5.8996086, -50.5714645, -5.9500079, -34.4623184, 34.5234451
40: -41.7083092, -3.3480577, -41.7134399, -3.3869538, -31.7886353, 31.8451881
41: -31.1700516, -4.1953392, -31.1703777, -4.2264099, -20.0409927, 20.0596676
42: -18.1290874, 2.5923553, -18.1258965, 2.5868864, -19.5404797, 19.5845146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=102, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1314
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1313
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_A2_A1_A2_B1

### Relational analysis result of IS_A2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8738397, upper bound: 17.8507256
time: 16.32 seconds

## Relational analysis of IS_A2_A2_A1_A2_B2

### Relational analysis result of IS_A2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8822135, upper bound: 17.8952011
time: 18.50 seconds

## BFS IS instance: IS_A2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -28.6093750, 5.8130612, -28.5895138, 5.8161907, -34.4255676, 34.4025764
1: -15.2285652, 11.1437092, -15.2083616, 11.1449032, -26.3734684, 26.3520699
2: -12.3386745, 10.9182634, -12.3142815, 10.9217415, -23.2604160, 23.2325439
3: -8.9838543, 15.7917776, -8.9691133, 15.8037901, -24.7876434, 24.7608910
4: -12.7297268, 13.2704172, -12.7244740, 13.2779922, -26.0077190, 25.9948921
5: -9.9289465, 18.0516376, -9.9131870, 18.0611610, -27.9901085, 27.9648247
6: -27.4579887, -2.9575086, -27.4633789, -2.9760532, -18.9725037, 18.9917297
7: -13.2382717, 17.7152367, -13.2340984, 17.7175770, -30.9558487, 30.9493351
8: -16.9784222, 15.8076363, -16.9671688, 15.8155107, -31.7382812, 31.7395706
9: -12.2281399, 13.5706539, -12.2208672, 13.5901947, -21.3613243, 21.3331604
10: -13.0836754, 24.7355919, -13.0399466, 24.7513580, -34.6552963, 34.5923691
11: -22.6966076, 12.7999077, -22.6939182, 12.8131199, -33.8418427, 33.8298607
12: -20.8438950, 15.4439192, -20.8539715, 15.4391184, -36.1152840, 36.1586075
13: -21.0511246, 11.3197775, -21.0461426, 11.3232470, -25.8418694, 25.8147545
14: -43.0257797, 3.4433789, -42.9910507, 3.4499154, -34.3582077, 34.3126717
15: -15.1326399, 9.8675327, -15.0911055, 9.8765612, -24.4717598, 24.4307022
16: -21.1196709, 13.1499443, -21.1072311, 13.1583328, -33.3249512, 33.3111191
17: -33.8439560, 27.5041924, -33.8283310, 27.5115585, -52.4426270, 52.4169312
18: -17.6614189, 7.9587297, -17.6680489, 7.9582343, -24.3665771, 24.3760586
19: -20.0911446, 2.0225391, -20.0980644, 2.0210061, -21.5009537, 21.5104485
20: -10.1522942, 10.2794495, -10.1661091, 10.2714348, -19.6895447, 19.7105064
21: -20.6858635, 7.2028189, -20.6958389, 7.1967025, -27.8825665, 27.8986588
22: -22.9040680, 9.3145943, -22.9194870, 9.3320465, -31.3163223, 31.3275986
23: -19.3573761, 4.2639976, -19.3600559, 4.2693491, -22.3310318, 22.3379173
24: -26.7253799, -1.7058663, -26.7459507, -1.7096033, -21.4382706, 21.4711876
25: -13.2903500, 9.4996805, -13.2964077, 9.4899492, -21.3181572, 21.3491173
26: -28.9129353, 8.7488956, -28.9307499, 8.7686729, -37.6424103, 37.6647720
27: -28.5691795, 0.3089004, -28.5890274, 0.3094807, -24.4606323, 24.4820480
28: -18.5238037, 6.3240852, -18.5375023, 6.3157139, -23.9583435, 23.9844818
29: -32.0567436, 5.0040588, -32.0724907, 5.0390816, -35.7375641, 35.7253036
30: -18.4728889, 8.4005394, -18.4911461, 8.3875961, -25.7242317, 25.7592773
31: -18.0057354, 8.5108299, -18.0100403, 8.5002670, -25.0878983, 25.1078129
32: -21.4068108, 4.2427168, -21.4121685, 4.2195778, -22.2689667, 22.3121910
33: -39.3179626, 1.1486597, -39.3208733, 1.0994968, -32.2302399, 32.2914124
34: -30.8226833, 2.2114673, -30.8208637, 2.1801162, -27.8152924, 27.8427048
35: -30.3057575, 2.5007887, -30.3088799, 2.4671311, -26.2337036, 26.2632484
36: -31.7536926, 0.2054210, -31.7567024, 0.1805758, -24.5300064, 24.5528488
37: -47.3470726, -6.5126133, -47.3591805, -6.5403948, -32.5794182, 32.6252136
38: -40.6553993, -2.1743650, -40.6634750, -2.1910114, -27.7470093, 27.7635307
39: -50.5523911, -5.9103622, -50.5601006, -5.9547510, -34.4335098, 34.4912262
40: -41.7033005, -3.3694229, -41.7096748, -3.3996544, -31.7711830, 31.8155785
41: -31.1572037, -4.2387652, -31.1638145, -4.2481203, -20.0065079, 20.0247135
42: -18.1258469, 2.5659175, -18.1256332, 2.5715733, -19.5140686, 19.5452957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=102, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1313
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A2_A2_A2_A1_A1

### Relational analysis result of IS_A2_A2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8667656, upper bound: 17.8742400
time: 20.89 seconds

## Relational analysis of IS_A2_A2_A2_A1_A2

### Relational analysis result of IS_A2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8526792, upper bound: 17.8885783
time: 25.74 seconds

## BFS IS instance: IS_A2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -28.6835632, 5.8229685, -28.6250839, 5.8190284, -34.5025902, 34.4480515
1: -15.2638540, 11.1529503, -15.2248678, 11.1480551, -26.4119091, 26.3778191
2: -12.3811398, 10.9284296, -12.3343639, 10.9246435, -23.3057823, 23.2627945
3: -9.0351124, 15.8146830, -8.9934177, 15.8106804, -24.8457928, 24.8081017
4: -12.7662334, 13.2871857, -12.7417240, 13.2834291, -26.0496635, 26.0289097
5: -9.9741917, 18.0672073, -9.9345970, 18.0650978, -28.0392895, 28.0018044
6: -27.4759750, -2.9149084, -27.4713612, -2.9554472, -19.0130501, 19.0321598
7: -13.2802162, 17.7262535, -13.2536764, 17.7225342, -31.0027504, 30.9799309
8: -17.0332623, 15.8252220, -16.9929943, 15.8222294, -31.8029175, 31.7993393
9: -12.3000507, 13.5991955, -12.2554407, 13.5963411, -21.4117889, 21.3974705
10: -13.1592026, 24.7666359, -13.0754938, 24.7597828, -34.7219009, 34.6590500
11: -22.7276859, 12.8687057, -22.7010078, 12.8460903, -33.9067535, 33.8992577
12: -20.8695564, 15.4609261, -20.8658714, 15.4460087, -36.1798401, 36.1966209
13: -21.1284370, 11.3518677, -21.0834293, 11.3304110, -25.8941956, 25.8854027
14: -43.1227531, 3.4584970, -43.0375061, 3.4541817, -34.4339180, 34.3750458
15: -15.1633806, 9.8851147, -15.1058121, 9.8838530, -24.5118141, 24.4687233
16: -21.1884937, 13.1665382, -21.1393833, 13.1626492, -33.3863144, 33.3595505
17: -33.9151955, 27.5206623, -33.8596039, 27.5194969, -52.5296326, 52.4694061
18: -17.6912556, 8.0237198, -17.6753349, 7.9894948, -24.4289246, 24.4234982
19: -20.1159897, 2.0827618, -20.1028137, 2.0501304, -21.5543137, 21.5596771
20: -10.1730442, 10.3418808, -10.1698322, 10.3015385, -19.7416687, 19.7713318
21: -20.7157097, 7.2731285, -20.7018833, 7.2306137, -27.9463234, 27.9750118
22: -22.9353199, 9.4036255, -22.9263992, 9.3751450, -31.3906937, 31.4210815
23: -19.3825607, 4.3181610, -19.3650608, 4.2954998, -22.3822708, 22.3890877
24: -26.7602539, -1.6239977, -26.7539730, -1.6700010, -21.5126953, 21.5325165
25: -13.3121958, 9.5609245, -13.3015280, 9.5196381, -21.3699341, 21.4147072
26: -28.9446831, 8.8414230, -28.9362106, 8.8132286, -37.7170029, 37.7439499
27: -28.6034966, 0.3970752, -28.5949612, 0.3520083, -24.5363464, 24.5439644
28: -18.5464172, 6.3875551, -18.5406036, 6.3463159, -24.0110855, 24.0417862
29: -32.0969086, 5.1079865, -32.0827141, 5.0892715, -35.8281021, 35.8372116
30: -18.5016479, 8.4701748, -18.4974556, 8.4211407, -25.7872429, 25.8297119
31: -18.0371876, 8.5675316, -18.0187340, 8.5276251, -25.1474228, 25.1607361
32: -21.4257832, 4.2715225, -21.4194088, 4.2330647, -22.3133545, 22.3545761
33: -39.3385468, 1.1794000, -39.3298836, 1.1138659, -32.2670593, 32.3336372
34: -30.8362770, 2.2424402, -30.8274593, 2.1943154, -27.8390656, 27.8853607
35: -30.3238983, 2.5235476, -30.3169880, 2.4775152, -26.2637253, 26.2972679
36: -31.7710247, 0.2565722, -31.7640953, 0.2054336, -24.5736084, 24.6059227
37: -47.3785210, -6.4551678, -47.3717041, -6.5129580, -32.6443176, 32.6816750
38: -40.6786804, -2.1190319, -40.6717949, -2.1651592, -27.7895927, 27.8173847
39: -50.5794144, -5.8922668, -50.5721931, -5.9468203, -34.4703979, 34.5277634
40: -41.7221375, -3.3380895, -41.7188568, -3.3851581, -31.8018951, 31.8498077
41: -31.1787262, -4.1884613, -31.1725578, -4.2237320, -20.0574036, 20.0705681
42: -18.1407051, 2.6005163, -18.1308651, 2.5880241, -19.5545902, 19.5866089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=102, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1313
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8465534, upper bound: 17.8518497
time: 27.07 seconds

## Relational analysis of IS_A2_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8961993, upper bound: 17.8961990
time: 26.10 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 55.67 seconds
IS_A2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 55.67
Output dim: 10, lower bound: -17.8879247, upper bound: 17.8419849
IS_A2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 55.67
Output dim: 10, lower bound: -17.8961993, upper bound: 17.8863218
IS_A2_A2_A1_A1_A1, status: Status.VERIFIED, split count: 5, time: 55.67
Output dim: 10, lower bound: -17.8526792, upper bound: 17.8731976
IS_A2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 55.67
Output dim: 10, lower bound: -17.8526792, upper bound: 17.8875334
IS_A2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 5, time: 55.67
Output dim: 10, lower bound: -17.8738397, upper bound: 17.8507256
IS_A2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 55.67
Output dim: 10, lower bound: -17.8822135, upper bound: 17.8952011
IS_A2_A2_A2_A1_A1, status: Status.VERIFIED, split count: 5, time: 55.67
Output dim: 10, lower bound: -17.8667656, upper bound: 17.8742400
IS_A2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 55.67
Output dim: 10, lower bound: -17.8526792, upper bound: 17.8885783
IS_A2_A2_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 55.67
Output dim: 10, lower bound: -17.8465534, upper bound: 17.8518497
IS_A2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 55.67
Output dim: 10, lower bound: -17.8961993, upper bound: 17.8961990

## BFS IS instance: IS_A2_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -28.6063156, 5.8076448, -28.4719391, 5.7979417, -34.4042587, 34.2795830
1: -15.2147942, 11.1352091, -15.1488533, 11.1274662, -26.3422604, 26.2840614
2: -12.3363552, 10.9160280, -12.2507172, 10.9070110, -23.2433662, 23.1667442
3: -8.9937496, 15.8020039, -8.8984604, 15.7857132, -24.7794628, 24.7004642
4: -12.7334795, 13.2764616, -12.6651211, 13.2604923, -25.9939728, 25.9415817
5: -9.9363470, 18.0594406, -9.8441753, 18.0447903, -27.9811363, 27.9036160
6: -27.4574394, -2.9756908, -27.4426651, -3.0703859, -18.8835850, 18.9455948
7: -13.2355289, 17.7111740, -13.1712093, 17.7003288, -30.9358578, 30.8823833
8: -16.9788933, 15.8114815, -16.8751488, 15.7998037, -31.7344894, 31.6589355
9: -12.2402754, 13.5876999, -12.1176920, 13.5685606, -21.3192520, 21.2468834
10: -13.0774021, 24.7395363, -12.9355640, 24.7177792, -34.6030502, 34.4957428
11: -22.6937237, 12.8201504, -22.6764526, 12.7477493, -33.7723770, 33.8261337
12: -20.8504944, 15.4362612, -20.8437729, 15.4029284, -36.0984154, 36.1227875
13: -21.0734482, 11.3251972, -20.9646301, 11.3017073, -25.8104401, 25.7465019
14: -43.0231094, 3.4349251, -42.8616180, 3.4260445, -34.3083038, 34.1854401
15: -15.1146946, 9.8692150, -15.0231476, 9.8534727, -24.4405746, 24.3628197
16: -21.1187096, 13.1480427, -21.0327511, 13.1370811, -33.2864990, 33.2345734
17: -33.8457718, 27.5027103, -33.7431297, 27.4919701, -52.4255219, 52.3274918
18: -17.6676273, 7.9730344, -17.6497498, 7.8807645, -24.2915878, 24.3461266
19: -20.0964451, 2.0426738, -20.0795135, 1.9500538, -21.4363823, 21.4967117
20: -10.1634941, 10.2950687, -10.1477089, 10.1925755, -19.6201859, 19.6996155
21: -20.6933174, 7.2226863, -20.6724072, 7.1070981, -27.8004150, 27.8950939
22: -22.9191093, 9.3482780, -22.8985710, 9.2416153, -31.2423096, 31.3300705
23: -19.3591003, 4.2816405, -19.3436184, 4.2140312, -22.2763443, 22.3320389
24: -26.7449074, -1.6818037, -26.7258644, -1.8103261, -21.3580780, 21.4450684
25: -13.2939215, 9.5151739, -13.2772846, 9.4099779, -21.2458229, 21.3427086
26: -28.9299259, 8.7829266, -28.9077492, 8.6719923, -37.5679855, 37.6425400
27: -28.5898762, 0.3382111, -28.5667572, 0.2079682, -24.3822899, 24.4570503
28: -18.5346889, 6.3446140, -18.5193615, 6.2414761, -23.8974380, 23.9757881
29: -32.0753021, 5.0514612, -32.0496063, 4.9466648, -35.6646957, 35.7458344
30: -18.4876041, 8.4219275, -18.4691124, 8.3053026, -25.6541634, 25.7523804
31: -18.0098896, 8.5233650, -17.9871101, 8.4194241, -25.0134621, 25.0835495
32: -21.4014034, 4.2180772, -21.3879070, 4.1572437, -22.2091179, 22.2704697
33: -39.3091049, 1.1113186, -39.2939682, 1.0218501, -32.1477737, 32.2290955
34: -30.8082066, 2.1812496, -30.7985325, 2.1223860, -27.7471161, 27.8029251
35: -30.2954750, 2.4689660, -30.2861271, 2.4095693, -26.1686592, 26.2122421
36: -31.7432404, 0.1823490, -31.7298889, 0.0919130, -24.4371948, 24.5002899
37: -47.3506393, -6.5133495, -47.3259659, -6.6160297, -32.5088463, 32.5699806
38: -40.6526108, -2.1889167, -40.6321449, -2.2785025, -27.6484146, 27.7067604
39: -50.5436592, -5.9574881, -50.5281639, -6.0149727, -34.3679962, 34.4172592
40: -41.6962738, -3.3844543, -41.6844025, -3.4457583, -31.7167740, 31.7671013
41: -31.1594925, -4.2412519, -31.1435089, -4.3248267, -19.9378853, 19.9914665
42: -18.1289558, 2.5750680, -18.1120243, 2.5319624, -19.4834957, 19.5410156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=102, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1314
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A2_A1_A2_A2_B1_A1

### Relational analysis result of IS_A2_A1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8427064, upper bound: 17.8226764
time: 29.64 seconds

## Relational analysis of IS_A2_A1_A2_A2_B1_A2

### Relational analysis result of IS_A2_A1_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8841006, upper bound: 17.8381631
time: 18.37 seconds

## BFS IS instance: IS_A2_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -28.6651955, 5.8110452, -28.6223373, 5.8154178, -34.4806137, 34.4333839
1: -15.2441759, 11.1393585, -15.2232494, 11.1437845, -26.3879604, 26.3626080
2: -12.3684511, 10.9197731, -12.3328705, 10.9221287, -23.2905807, 23.2526436
3: -9.0298271, 15.8087225, -8.9916897, 15.8096218, -24.8394489, 24.8004112
4: -12.7626667, 13.2827291, -12.7403688, 13.2824783, -26.0451450, 26.0230980
5: -9.9707003, 18.0645275, -9.9329529, 18.0644836, -28.0351830, 27.9974804
6: -27.4643421, -2.9314814, -27.4680290, -2.9570112, -18.9622135, 19.0134735
7: -13.2672482, 17.7168407, -13.2518663, 17.7195244, -30.9867725, 30.9687080
8: -17.0235348, 15.8164806, -16.9912605, 15.8197422, -31.8009109, 31.7844734
9: -12.2924700, 13.5940828, -12.2534409, 13.5952749, -21.3991241, 21.3461590
10: -13.1302729, 24.7498074, -13.0730886, 24.7541084, -34.6914825, 34.6113129
11: -22.7009144, 12.8563614, -22.6995468, 12.8415651, -33.8662186, 33.8856430
12: -20.8591423, 15.4525423, -20.8622475, 15.4441547, -36.1691170, 36.2150269
13: -21.1176548, 11.3327293, -21.0788784, 11.3295660, -25.8827820, 25.8316917
14: -43.0896530, 3.4402695, -43.0336876, 3.4476995, -34.3953781, 34.3106575
15: -15.1462107, 9.8773050, -15.1043205, 9.8817825, -24.5009956, 24.4498901
16: -21.1596413, 13.1537724, -21.1367264, 13.1581211, -33.3509369, 33.3260040
17: -33.8903770, 27.5108719, -33.8555145, 27.5163345, -52.4996796, 52.4210052
18: -17.6742020, 8.0133905, -17.6739044, 7.9857359, -24.3690262, 24.4119911
19: -20.1022873, 2.0805025, -20.1014996, 2.0486031, -21.5197411, 21.5551949
20: -10.1685047, 10.3365870, -10.1686831, 10.2996607, -19.7202148, 19.7628632
21: -20.7007370, 7.2692852, -20.7001743, 7.2282619, -27.9289989, 27.9694595
22: -22.9265823, 9.3990707, -22.9249229, 9.3730984, -31.3776169, 31.4069290
23: -19.3643322, 4.3119235, -19.3637161, 4.2929444, -22.3557625, 22.3820648
24: -26.7517014, -1.6283617, -26.7527447, -1.6719742, -21.4663010, 21.5251160
25: -13.3000212, 9.5568686, -13.3001804, 9.5176802, -21.3595047, 21.4061089
26: -28.9363880, 8.8366413, -28.9345875, 8.8108616, -37.6929703, 37.7227859
27: -28.5962067, 0.3927712, -28.5942822, 0.3500814, -24.4864120, 24.5367889
28: -18.5389824, 6.3843360, -18.5393410, 6.3445807, -23.9927444, 24.0343475
29: -32.0854149, 5.1056957, -32.0811958, 5.0873823, -35.8133392, 35.8311462
30: -18.4944649, 8.4658051, -18.4958858, 8.4192944, -25.7675552, 25.8232307
31: -18.0184975, 8.5642815, -18.0171585, 8.5258017, -25.1109695, 25.1542244
32: -21.4093437, 4.2467122, -21.4138050, 4.2314482, -22.2974052, 22.3245049
33: -39.3189278, 1.1467237, -39.3236847, 1.1122823, -32.2314072, 32.2932930
34: -30.8155670, 2.2092748, -30.8207664, 2.1927562, -27.7961426, 27.8455734
35: -30.3037281, 2.4952207, -30.3100376, 2.4763236, -26.2282257, 26.2617340
36: -31.7515507, 0.2259364, -31.7572403, 0.2035911, -24.5267639, 24.5683708
37: -47.3644371, -6.4741111, -47.3681488, -6.5146031, -32.5972824, 32.6548691
38: -40.6623383, -2.1452227, -40.6659622, -2.1674061, -27.7172012, 27.7853050
39: -50.5559998, -5.9308124, -50.5642166, -5.9481492, -34.4282303, 34.4802742
40: -41.7056351, -3.3608055, -41.7134476, -3.3862867, -31.7598953, 31.8211174
41: -31.1675301, -4.2024341, -31.1690350, -4.2250700, -20.0080986, 20.0546246
42: -18.1347752, 2.5966158, -18.1300278, 2.5872946, -19.5476799, 19.5871391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=102, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A2_A1_A2_A2_B2_A1

### Relational analysis result of IS_A2_A1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8924435, upper bound: 17.8682312
time: 30.09 seconds

## Relational analysis of IS_A2_A1_A2_A2_B2_A2

### Relational analysis result of IS_A2_A1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8924435, upper bound: 17.8825841
time: 17.74 seconds

## BFS IS instance: IS_A2_A2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -28.6388741, 5.7900629, -28.5770912, 5.8139901, -34.4528656, 34.3671532
1: -15.2485867, 11.1212101, -15.1978979, 11.1425209, -26.3911076, 26.3191071
2: -12.3546801, 10.9057779, -12.3086548, 10.9196758, -23.2743568, 23.2144318
3: -8.9990845, 15.7865448, -8.9645510, 15.8021059, -24.8011894, 24.7510948
4: -12.7350712, 13.2556934, -12.7167473, 13.2762890, -26.0113602, 25.9724407
5: -9.9532995, 18.0430126, -9.9090805, 18.0598507, -28.0131493, 27.9520931
6: -27.4566193, -2.9411917, -27.4598312, -2.9775548, -18.9454823, 19.0108757
7: -13.2607632, 17.6992607, -13.2256804, 17.7154655, -30.9762287, 30.9249420
8: -16.9920769, 15.7914133, -16.9576054, 15.8124409, -31.7625504, 31.7127838
9: -12.2407265, 13.5645618, -12.2151184, 13.5890465, -21.3783493, 21.3188381
10: -13.1181488, 24.7147732, -13.0277376, 24.7484665, -34.6877441, 34.5584946
11: -22.7171478, 12.7985020, -22.6896496, 12.8107243, -33.8625031, 33.8191986
12: -20.8423500, 15.4410667, -20.8511620, 15.4368687, -36.1145935, 36.1571121
13: -21.0448227, 11.3133488, -21.0428791, 11.3170471, -25.8455887, 25.8077774
14: -43.0592079, 3.4260755, -42.9820061, 3.4473562, -34.3977356, 34.2862282
15: -15.1410599, 9.8656139, -15.0867348, 9.8742161, -24.4796448, 24.4237938
16: -21.1467285, 13.1318607, -21.0937901, 13.1561403, -33.3617859, 33.2794571
17: -33.9255142, 27.4964542, -33.8232727, 27.5085373, -52.5218506, 52.3980103
18: -17.6544685, 7.9515357, -17.6628590, 7.9547930, -24.3597679, 24.3647957
19: -20.0829105, 2.0152903, -20.0957203, 2.0146885, -21.4852066, 21.5023880
20: -10.1480122, 10.3078318, -10.1635647, 10.2691898, -19.6725159, 19.7379799
21: -20.6855221, 7.2068892, -20.6930428, 7.1954374, -27.8809586, 27.8999329
22: -22.9044266, 9.3267584, -22.9165115, 9.3291674, -31.3133545, 31.3390350
23: -19.3507767, 4.2518353, -19.3585968, 4.2611227, -22.3134041, 22.3382874
24: -26.7353859, -1.6887312, -26.7462616, -1.7114620, -21.4424515, 21.4846916
25: -13.2916651, 9.5081472, -13.2960730, 9.4871750, -21.3127556, 21.3627167
26: -28.8995628, 8.7522783, -28.9277840, 8.7616940, -37.6163101, 37.6734314
27: -28.5694065, 0.3366165, -28.5868778, 0.3082047, -24.4470291, 24.5099564
28: -18.5110149, 6.3374529, -18.5357609, 6.3086333, -23.9382553, 23.9976234
29: -32.0640068, 5.0048265, -32.0691986, 5.0382414, -35.7385178, 35.7277908
30: -18.4732609, 8.4038792, -18.4882469, 8.3852386, -25.7135315, 25.7740784
31: -18.0074348, 8.5145721, -18.0074177, 8.4963112, -25.0843887, 25.1086578
32: -21.4044437, 4.2854228, -21.4089222, 4.2180457, -22.2598991, 22.3536148
33: -39.3173447, 1.2026644, -39.3188972, 1.0957255, -32.2252121, 32.3438072
34: -30.8209782, 2.2621870, -30.8179932, 2.1766605, -27.8093414, 27.8929558
35: -30.2939110, 2.5467582, -30.3064957, 2.4595447, -26.2132950, 26.3162270
36: -31.7277985, 0.2516983, -31.7531433, 0.1683688, -24.4910583, 24.6071129
37: -47.3454437, -6.4774294, -47.3563118, -6.5426936, -32.5728149, 32.6580963
38: -40.6347504, -2.1145535, -40.6596184, -2.1970630, -27.7165680, 27.8318119
39: -50.5516129, -5.8490515, -50.5580482, -5.9580846, -34.4281540, 34.5541115
40: -41.6932678, -3.3344193, -41.7032166, -3.4015703, -31.7614136, 31.8554230
41: -31.1547298, -4.2195339, -31.1610336, -4.2510023, -19.9958782, 20.0396652
42: -18.1180630, 2.5600057, -18.1205254, 2.5686183, -19.5030079, 19.5483093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=101, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 731

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_A2_A1_A1_A2_B1

### Relational analysis result of IS_A2_A2_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8427064, upper bound: 17.8420223
time: 19.93 seconds

## Relational analysis of IS_A2_A2_A1_A1_A2_B2

### Relational analysis result of IS_A2_A2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8515448, upper bound: 17.8864975
time: 19.94 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -28.6556721, 5.7986865, -28.6106873, 5.8175659, -34.4732361, 34.4093742
1: -15.2420254, 11.1297550, -15.2133102, 11.1464128, -26.3884392, 26.3430653
2: -12.3689728, 10.9146557, -12.3276577, 10.9230461, -23.2920189, 23.2423134
3: -9.0251408, 15.8047495, -8.9877672, 15.8091183, -24.8342590, 24.7925167
4: -12.7501373, 13.2692890, -12.7330475, 13.2819729, -26.0321102, 26.0023365
5: -9.9650097, 18.0578537, -9.9293823, 18.0641975, -28.0292072, 27.9872360
6: -27.4685020, -2.9209294, -27.4680614, -2.9582415, -18.9397469, 19.0316887
7: -13.2619362, 17.7084198, -13.2442379, 17.7210541, -30.9829903, 30.9526577
8: -17.0134830, 15.8020077, -16.9819927, 15.8197689, -31.7940826, 31.7643738
9: -12.2869949, 13.5907011, -12.2480278, 13.5953388, -21.4040680, 21.3358402
10: -13.1320477, 24.7434120, -13.0615730, 24.7576828, -34.6941223, 34.5898209
11: -22.7168579, 12.8641481, -22.6967125, 12.8440914, -33.8867416, 33.8856621
12: -20.8615284, 15.4500980, -20.8638916, 15.4427729, -36.1704483, 36.2127838
13: -21.1151924, 11.3381729, -21.0800362, 11.3241997, -25.8809814, 25.8346710
14: -43.0994263, 3.4400949, -43.0264778, 3.4522209, -34.4181137, 34.2967110
15: -15.1523228, 9.8786316, -15.1003876, 9.8816423, -24.4999199, 24.4511795
16: -21.1595573, 13.1466532, -21.1246624, 13.1611557, -33.3673172, 33.3070984
17: -33.8985901, 27.5096188, -33.8531647, 27.5177994, -52.5106277, 52.4117813
18: -17.6787033, 8.0123081, -17.6705017, 7.9860053, -24.3743324, 24.4065475
19: -20.1022358, 2.0700946, -20.1003914, 2.0432014, -21.5137482, 21.5467110
20: -10.1655846, 10.3363400, -10.1677771, 10.2978859, -19.7053604, 19.7653809
21: -20.7072563, 7.2704010, -20.6992416, 7.2278485, -27.9351044, 27.9696426
22: -22.9260445, 9.3969707, -22.9236279, 9.3707714, -31.3734131, 31.4125748
23: -19.3697319, 4.3048077, -19.3635750, 4.2883282, -22.3560715, 22.3805618
24: -26.7592583, -1.6265397, -26.7542133, -1.6735916, -21.4681625, 21.5258217
25: -13.3091030, 9.5549288, -13.3012857, 9.5156898, -21.3594513, 21.4083862
26: -28.9265175, 8.8257856, -28.9336128, 8.8044500, -37.6671753, 37.7260590
27: -28.5982704, 0.3931055, -28.5933495, 0.3489871, -24.4729996, 24.5405197
28: -18.5316048, 6.3730564, -18.5393276, 6.3380070, -23.9762039, 24.0273895
29: -32.0863838, 5.1055632, -32.0791779, 5.0868626, -35.8110352, 35.8320694
30: -18.4944973, 8.4669123, -18.4955463, 8.4177322, -25.7638321, 25.8256226
31: -18.0289936, 8.5575466, -18.0159569, 8.5223198, -25.1161690, 25.1481190
32: -21.4181366, 4.2669172, -21.4168968, 4.2307606, -22.2980957, 22.3476601
33: -39.3344002, 1.1710110, -39.3286362, 1.1089816, -32.2415771, 32.3243790
34: -30.8291111, 2.2344322, -30.8254700, 2.1898365, -27.8045235, 27.8778419
35: -30.3067665, 2.5076399, -30.3155079, 2.4690614, -26.2219582, 26.2891312
36: -31.7426376, 0.2313888, -31.7616138, 0.1917181, -24.5033188, 24.5891647
37: -47.3694458, -6.4603734, -47.3691406, -6.5164871, -32.5981598, 32.6743927
38: -40.6568985, -2.1327791, -40.6692200, -2.1726718, -27.7012634, 27.8102303
39: -50.5746155, -5.8996086, -50.5709839, -5.9510098, -34.4409637, 34.5228958
40: -41.7083092, -3.3480577, -41.7131500, -3.3879304, -31.7614136, 31.8448372
41: -31.1700516, -4.1953392, -31.1699562, -4.2274084, -19.9995346, 20.0592556
42: -18.1290874, 2.5923553, -18.1256371, 2.5865054, -19.5395527, 19.5898323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=102, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A2_A2_A1_A2_B2_A1

### Relational analysis result of IS_A2_A2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8427064, upper bound: 17.8721684
time: 20.95 seconds

## Relational analysis of IS_A2_A2_A1_A2_B2_A2

### Relational analysis result of IS_A2_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8784575, upper bound: 17.8914388
time: 26.04 seconds

## BFS IS instance: IS_A2_A2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -28.6667595, 5.8143406, -28.5892391, 5.8152728, -34.4820328, 34.4035797
1: -15.2704239, 11.1444283, -15.2082481, 11.1440067, -26.4144306, 26.3526764
2: -12.3668270, 10.9195404, -12.3141785, 10.9211855, -23.2880135, 23.2337189
3: -9.0090647, 15.7964687, -8.9689865, 15.8033419, -24.8124065, 24.7654552
4: -12.7511444, 13.2735901, -12.7243805, 13.2775249, -26.0286694, 25.9979706
5: -9.9624920, 18.0523834, -9.9130545, 18.0605736, -28.0230656, 27.9654388
6: -27.4640884, -2.9351478, -27.4627609, -2.9762282, -18.9778137, 19.0110016
7: -13.2790165, 17.7170715, -13.2339001, 17.7167435, -30.9957600, 30.9509716
8: -17.0118275, 15.8146400, -16.9670734, 15.8144608, -31.7712250, 31.7462006
9: -12.2537823, 13.5730705, -12.2207670, 13.5897484, -21.3857918, 21.3354645
10: -13.1453142, 24.7379913, -13.0397100, 24.7502995, -34.7152176, 34.5933456
11: -22.7279739, 12.8030777, -22.6934776, 12.8119888, -33.8734856, 33.8323975
12: -20.8504181, 15.4518862, -20.8526802, 15.4387569, -36.1227722, 36.1695099
13: -21.0581074, 11.3270664, -21.0446224, 11.3229179, -25.8583908, 25.8171120
14: -43.0825233, 3.4444866, -42.9906731, 3.4491253, -34.4133148, 34.3133621
15: -15.1521187, 9.8721008, -15.0908813, 9.8760910, -24.4911728, 24.4341049
16: -21.1756592, 13.1517448, -21.1068611, 13.1573772, -33.3805084, 33.3122253
17: -33.9420891, 27.5074902, -33.8280182, 27.5099087, -52.5393524, 52.4194412
18: -17.6670094, 7.9629564, -17.6673412, 7.9569898, -24.3753891, 24.3813820
19: -20.0966759, 2.0279670, -20.0978546, 2.0203118, -21.5046883, 21.5150986
20: -10.1554680, 10.3133917, -10.1653786, 10.2713757, -19.6918602, 19.7437096
21: -20.6939621, 7.2096119, -20.6953373, 7.1966238, -27.8905869, 27.9049492
22: -22.9136600, 9.3334198, -22.9188480, 9.3318863, -31.3248138, 31.3471069
23: -19.3635921, 4.2651868, -19.3597794, 4.2672400, -22.3321228, 22.3465080
24: -26.7363853, -1.6861405, -26.7454872, -1.7096648, -21.4477425, 21.4908829
25: -13.2947598, 9.5141172, -13.2960386, 9.4897470, -21.3191452, 21.3687973
26: -28.9177418, 8.7679253, -28.9300728, 8.7685566, -37.6436234, 37.6910248
27: -28.5746498, 0.3405566, -28.5882263, 0.3093886, -24.4654007, 24.5131645
28: -18.5258369, 6.3519697, -18.5368805, 6.3155665, -23.9600754, 24.0118446
29: -32.0745163, 5.0072460, -32.0721207, 5.0388794, -35.7498932, 35.7323685
30: -18.4803772, 8.4071217, -18.4897995, 8.3871908, -25.7241592, 25.7778091
31: -18.0156441, 8.5245686, -18.0096703, 8.5001764, -25.0965729, 25.1206989
32: -21.4120827, 4.2900023, -21.4110909, 4.2192478, -22.2721596, 22.3601837
33: -39.3214798, 1.2110615, -39.3196793, 1.0993590, -32.2326279, 32.3526459
34: -30.8281460, 2.2701874, -30.8196831, 2.1800404, -27.8193359, 27.9001236
35: -30.3110256, 2.5626426, -30.3076286, 2.4670472, -26.2381096, 26.3239899
36: -31.7561512, 0.2768393, -31.7553596, 0.1805100, -24.5304222, 24.6235886
37: -47.3545113, -6.4722614, -47.3582993, -6.5405245, -32.5860062, 32.6647453
38: -40.6565781, -2.1008105, -40.6619072, -2.1912470, -27.7466888, 27.8385868
39: -50.5563660, -5.8417311, -50.5587349, -5.9549174, -34.4362335, 34.5584373
40: -41.7071152, -3.3244801, -41.7086601, -3.3997302, -31.7746582, 31.8600159
41: -31.1634121, -4.2126489, -31.1632137, -4.2483201, -20.0122757, 20.0505714
42: -18.1296883, 2.5681701, -18.1254902, 2.5697632, -19.5171165, 19.5503922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=101, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 731

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A2_A2_A2_A1_A2_B1

### Relational analysis result of IS_A2_A2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8568471, upper bound: 17.8431454
time: 31.56 seconds

## Relational analysis of IS_A2_A2_A2_A1_A2_B2

### Relational analysis result of IS_A2_A2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8427064, upper bound: 17.8875396
time: 20.13 seconds

## BFS IS instance: IS_A2_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -28.6835632, 5.8229685, -28.6228008, 5.8188758, -34.5024376, 34.4457703
1: -15.2638540, 11.1529503, -15.2236519, 11.1478930, -26.4117470, 26.3766022
2: -12.3811398, 10.9284296, -12.3331766, 10.9245586, -23.3056984, 23.2616062
3: -9.0351124, 15.8146830, -8.9922075, 15.8103352, -24.8454475, 24.8068905
4: -12.7662334, 13.2871857, -12.7406721, 13.2832060, -26.0494385, 26.0278587
5: -9.9741917, 18.0672073, -9.9333782, 18.0649223, -28.0391140, 28.0005856
6: -27.4759750, -2.9149084, -27.4709969, -2.9569030, -18.9720783, 19.0318127
7: -13.2802162, 17.7262535, -13.2524481, 17.7223473, -31.0025635, 30.9787025
8: -17.0332623, 15.8252220, -16.9914742, 15.8218136, -31.8027954, 31.7977676
9: -12.3000507, 13.5991955, -12.2536983, 13.5960550, -21.4115028, 21.3524551
10: -13.1592026, 24.7666359, -13.0735884, 24.7594833, -34.7215805, 34.6246796
11: -22.7276859, 12.8687057, -22.7005692, 12.8453331, -33.8976974, 33.8988419
12: -20.8695564, 15.4609261, -20.8653908, 15.4446793, -36.1786423, 36.2251816
13: -21.1284370, 11.3518677, -21.0817642, 11.3300667, -25.8938065, 25.8440399
14: -43.1227531, 3.4584970, -43.0351410, 3.4540310, -34.4337044, 34.3238144
15: -15.1633806, 9.8851147, -15.1045580, 9.8835230, -24.5114784, 24.4614944
16: -21.1884937, 13.1665382, -21.1377373, 13.1624184, -33.3860550, 33.3398590
17: -33.9151955, 27.5206623, -33.8579140, 27.5191574, -52.5281219, 52.4331512
18: -17.6912556, 8.0237198, -17.6749916, 7.9882030, -24.3899384, 24.4231625
19: -20.1159897, 2.0827618, -20.1025352, 2.0488570, -21.5332375, 21.5594101
20: -10.1730442, 10.3418808, -10.1696014, 10.3000660, -19.7246895, 19.7711449
21: -20.7157097, 7.2731285, -20.7015610, 7.2290344, -27.9447441, 27.9746895
22: -22.9353199, 9.4036255, -22.9259682, 9.3734932, -31.3848877, 31.4206772
23: -19.3825607, 4.3181610, -19.3647518, 4.2944527, -22.3747864, 22.3887978
24: -26.7602539, -1.6239977, -26.7534561, -1.6717777, -21.4734802, 21.5320053
25: -13.3121958, 9.5609245, -13.3012447, 9.5182686, -21.3658676, 21.4144478
26: -28.9446831, 8.8414230, -28.9359169, 8.8113136, -37.6944962, 37.7436371
27: -28.6034966, 0.3970752, -28.5947037, 0.3501663, -24.4913292, 24.5436974
28: -18.5464172, 6.3875551, -18.5404358, 6.3449273, -23.9980240, 24.0416260
29: -32.0969086, 5.1079865, -32.0821075, 5.0875425, -35.8224411, 35.8365784
30: -18.5016479, 8.4701748, -18.4970932, 8.4196815, -25.7744560, 25.8293495
31: -18.0371876, 8.5675316, -18.0181961, 8.5261860, -25.1283798, 25.1602039
32: -21.4257832, 4.2715225, -21.4190540, 4.2319822, -22.3103409, 22.3542519
33: -39.3385468, 1.1794000, -39.3294563, 1.1125994, -32.2489929, 32.3332214
34: -30.8362770, 2.2424402, -30.8271236, 2.1931939, -27.8145027, 27.8850212
35: -30.3238983, 2.5235476, -30.3166370, 2.4765439, -26.2468185, 26.2968903
36: -31.7710247, 0.2565722, -31.7638054, 0.2038534, -24.5426788, 24.6056175
37: -47.3785210, -6.4551678, -47.3711319, -6.5143309, -32.6113434, 32.6810608
38: -40.6786804, -2.1190319, -40.6714935, -2.1668644, -27.7313499, 27.8170052
39: -50.5794144, -5.8922668, -50.5716743, -5.9478269, -34.4490051, 34.5272331
40: -41.7221375, -3.3380895, -41.7185593, -3.3861356, -31.7746887, 31.8494492
41: -31.1787262, -4.1884613, -31.1721458, -4.2247300, -20.0159435, 20.0701599
42: -18.1407051, 2.6005163, -18.1306057, 2.5876555, -19.5536575, 19.5919380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=102, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A2_A2_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8924435, upper bound: 17.8780986
time: 29.48 seconds

## Relational analysis of IS_A2_A2_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8924435, upper bound: 17.8924430
time: 26.34 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 58.17 seconds
IS_A2_A1_A2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 58.17
Output dim: 10, lower bound: -17.8427064, upper bound: 17.8226764
IS_A2_A1_A2_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 58.17
Output dim: 10, lower bound: -17.8841006, upper bound: 17.8381631
IS_A2_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 58.17
Output dim: 10, lower bound: -17.8924435, upper bound: 17.8682312
IS_A2_A1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 58.17
Output dim: 10, lower bound: -17.8924435, upper bound: 17.8825841
IS_A2_A2_A1_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 58.17
Output dim: 10, lower bound: -17.8427064, upper bound: 17.8420223
IS_A2_A2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 58.17
Output dim: 10, lower bound: -17.8515448, upper bound: 17.8864975
IS_A2_A2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 58.17
Output dim: 10, lower bound: -17.8427064, upper bound: 17.8721684
IS_A2_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 58.17
Output dim: 10, lower bound: -17.8784575, upper bound: 17.8914388
IS_A2_A2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 58.17
Output dim: 10, lower bound: -17.8568471, upper bound: 17.8431454
IS_A2_A2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 58.17
Output dim: 10, lower bound: -17.8427064, upper bound: 17.8875396
IS_A2_A2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 58.17
Output dim: 10, lower bound: -17.8924435, upper bound: 17.8780986
IS_A2_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 58.17
Output dim: 10, lower bound: -17.8924435, upper bound: 17.8924430

## BFS IS instance: IS_A2_A1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -28.6633530, 5.8002801, -28.6220093, 5.8135891, -34.4769440, 34.4222908
1: -15.2429209, 11.1288109, -15.2230244, 11.1419935, -26.3849144, 26.3518353
2: -12.3675556, 10.9140797, -12.3327160, 10.9211483, -23.2887039, 23.2467957
3: -9.0290213, 15.8036079, -8.9915609, 15.8087330, -24.8377533, 24.7951698
4: -12.7619295, 13.2778988, -12.7402363, 13.2816467, -26.0435753, 26.0181351
5: -9.9699173, 18.0573273, -9.9328175, 18.0632458, -28.0331631, 27.9901447
6: -27.4571438, -2.9322267, -27.4668159, -2.9571390, -18.9549675, 19.0112419
7: -13.2655048, 17.7067986, -13.2515526, 17.7178249, -30.9833298, 30.9583511
8: -17.0226688, 15.8056622, -16.9910927, 15.8178940, -31.7983093, 31.7736130
9: -12.2919006, 13.5893021, -12.2533522, 13.5944653, -21.3976288, 21.3412876
10: -13.1286964, 24.7374096, -13.0728168, 24.7519684, -34.6876678, 34.5986137
11: -22.6984959, 12.8476887, -22.6991196, 12.8400249, -33.8620911, 33.8758965
12: -20.8558445, 15.4512520, -20.8616600, 15.4439411, -36.1645432, 36.2131538
13: -21.1163921, 11.3316097, -21.0786514, 11.3293819, -25.8812408, 25.8287659
14: -43.0882874, 3.4314580, -43.0334549, 3.4461851, -34.3923798, 34.3018074
15: -15.1449394, 9.8734493, -15.1040993, 9.8811226, -24.4988708, 24.4457664
16: -21.1576366, 13.1412601, -21.1363716, 13.1560087, -33.3469696, 33.3131943
17: -33.8890686, 27.4891529, -33.8552856, 27.5126457, -52.4947510, 52.3991089
18: -17.6721287, 8.0118294, -17.6735497, 7.9854741, -24.3666420, 24.4094887
19: -20.1011620, 2.0801148, -20.1012993, 2.0485239, -21.5180969, 21.5544205
20: -10.1610527, 10.3361340, -10.1673946, 10.2995892, -19.7126198, 19.7610931
21: -20.6976166, 7.2688437, -20.6996174, 7.2281837, -27.9258003, 27.9684601
22: -22.9192562, 9.3979883, -22.9236794, 9.3729286, -31.3700638, 31.4046173
23: -19.3632393, 4.3111334, -19.3635387, 4.2928095, -22.3530693, 22.3808098
24: -26.7495975, -1.6287251, -26.7523880, -1.6720352, -21.4637451, 21.5242157
25: -13.2986546, 9.5556784, -13.2999392, 9.5174809, -21.3566818, 21.4045906
26: -28.9322243, 8.8360910, -28.9338551, 8.8107872, -37.6874084, 37.7210617
27: -28.5889854, 0.3922973, -28.5930481, 0.3500004, -24.4790802, 24.5350151
28: -18.5341721, 6.3835216, -18.5385132, 6.3444424, -23.9877472, 24.0327148
29: -32.0829582, 5.1040487, -32.0807648, 5.0871029, -35.8082733, 35.8288879
30: -18.4921379, 8.4643974, -18.4954796, 8.4190426, -25.7612267, 25.8212700
31: -18.0156918, 8.5634089, -18.0166626, 8.5256538, -25.1074524, 25.1526661
32: -21.3956795, 4.2458334, -21.4114971, 4.2313099, -22.2837524, 22.3214951
33: -39.3052139, 1.1454635, -39.3213615, 1.1120648, -32.2174072, 32.2896576
34: -30.8018131, 2.2086096, -30.8184280, 2.1926332, -27.7822762, 27.8423920
35: -30.2893238, 2.4943743, -30.3075714, 2.4761639, -26.2137032, 26.2583961
36: -31.7340889, 0.2253077, -31.7542553, 0.2034497, -24.5091438, 24.5647163
37: -47.3555450, -6.4751511, -47.3666077, -6.5147753, -32.5881424, 32.6523438
38: -40.6431427, -2.1460838, -40.6626854, -2.1675429, -27.6978416, 27.7810497
39: -50.5394363, -5.9316339, -50.5614090, -5.9482856, -34.4115067, 34.4766464
40: -41.6944656, -3.3615732, -41.7115250, -3.3864169, -31.7485886, 31.8185806
41: -31.1601372, -4.2031083, -31.1677780, -4.2251863, -20.0008221, 20.0527744
42: -18.1334114, 2.5958228, -18.1297855, 2.5871487, -19.5460205, 19.5853176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=101, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_A1_A2_A2_B2_A1_A1

### Relational analysis result of IS_A2_A1_A2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8480135, upper bound: 17.8598913
time: 20.07 seconds

## Relational analysis of IS_A2_A1_A2_A2_B2_A1_A2

### Relational analysis result of IS_A2_A1_A2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8480135, upper bound: 17.8238006
time: 17.47 seconds

## BFS IS instance: IS_A2_A1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -28.7227211, 5.8123431, -28.6220627, 5.8144884, -34.5372086, 34.4344063
1: -15.2860489, 11.1400213, -15.2231464, 11.1428537, -26.4289017, 26.3631668
2: -12.3966265, 10.9210272, -12.3327503, 10.9215889, -23.3182144, 23.2537766
3: -9.0550508, 15.8133736, -8.9915695, 15.8091507, -24.8642006, 24.8049431
4: -12.7840977, 13.2859192, -12.7402411, 13.2820129, -26.0661106, 26.0261612
5: -10.0042686, 18.0652962, -9.9328442, 18.0638714, -28.0681400, 27.9981403
6: -27.4704075, -2.9091139, -27.4674168, -2.9571800, -18.9675140, 19.0327682
7: -13.3079662, 17.7186832, -13.2516460, 17.7186890, -31.0266552, 30.9703293
8: -17.0569344, 15.8234587, -16.9911346, 15.8187103, -31.8338699, 31.7910805
9: -12.3181257, 13.5964737, -12.2533245, 13.5948334, -21.4236069, 21.3484612
10: -13.1919279, 24.7521820, -13.0728340, 24.7529984, -34.7513657, 34.6122894
11: -22.7322578, 12.8595362, -22.6990662, 12.8404198, -33.8978195, 33.8881531
12: -20.8656197, 15.4605017, -20.8609428, 15.4437962, -36.1766129, 36.2259102
13: -21.1246300, 11.3399601, -21.0773392, 11.3292389, -25.8992767, 25.8340034
14: -43.1464653, 3.4414072, -43.0333557, 3.4469252, -34.4505310, 34.3113060
15: -15.1657143, 9.8818445, -15.1041069, 9.8812990, -24.5204163, 24.4532700
16: -21.2155991, 13.1555748, -21.1363297, 13.1571808, -33.4064713, 33.3270493
17: -33.9885521, 27.5141830, -33.8552208, 27.5146866, -52.5964203, 52.4235611
18: -17.6797905, 8.0176048, -17.6732006, 7.9844871, -24.3778229, 24.4173317
19: -20.1077900, 2.0859394, -20.1012840, 2.0479107, -21.5234909, 21.5598335
20: -10.1716757, 10.3705101, -10.1679325, 10.2995958, -19.7225571, 19.7961121
21: -20.7088394, 7.2760878, -20.6996708, 7.2281885, -27.9370270, 27.9757576
22: -22.9362278, 9.4178848, -22.9242764, 9.3729572, -31.3861160, 31.4264679
23: -19.3705368, 4.3131161, -19.3634071, 4.2908392, -22.3568344, 22.3906708
24: -26.7626648, -1.6086440, -26.7523041, -1.6720281, -21.4757309, 21.5448074
25: -13.3043766, 9.5713081, -13.2998228, 9.5174932, -21.3604774, 21.4258041
26: -28.9411774, 8.8557205, -28.9339066, 8.8107624, -37.6941986, 37.7490540
27: -28.6016426, 0.4244637, -28.5934982, 0.3499894, -24.4911499, 24.5678291
28: -18.5410156, 6.4121852, -18.5387154, 6.3444357, -23.9944916, 24.0617409
29: -32.1031570, 5.1088600, -32.0808182, 5.0871506, -35.8256607, 35.8381271
30: -18.5019226, 8.4724045, -18.4945335, 8.4188938, -25.7674751, 25.8417625
31: -18.0283947, 8.5779953, -18.0167637, 8.5257101, -25.1195908, 25.1671429
32: -21.4146061, 4.2940106, -21.4127083, 4.2311096, -22.3005753, 22.3725243
33: -39.3224754, 1.2091060, -39.3225174, 1.1121130, -32.2337799, 32.3544998
34: -30.8210068, 2.2679801, -30.8195419, 2.1926556, -27.8001747, 27.9029808
35: -30.3089638, 2.5570693, -30.3087940, 2.4762268, -26.2326050, 26.3224564
36: -31.7539883, 0.2973139, -31.7558880, 0.2034802, -24.5271683, 24.6390915
37: -47.3718910, -6.4337859, -47.3672676, -6.5147257, -32.6037903, 32.6943817
38: -40.6634789, -2.0716276, -40.6644135, -2.1676168, -27.7168961, 27.8603497
39: -50.5599289, -5.8622098, -50.5628586, -5.9482837, -34.4309616, 34.5474854
40: -41.7094574, -3.3158784, -41.7124023, -3.3864079, -31.7633591, 31.8655624
41: -31.1737480, -4.1762996, -31.1684303, -4.2252574, -20.0138664, 20.0804863
42: -18.1386242, 2.5988712, -18.1298790, 2.5854959, -19.5507412, 19.5922432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=101, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=212, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 731

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_A1_A2_A2_B2_A2_A1

### Relational analysis result of IS_A2_A1_A2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8125182, upper bound: 17.8742360
time: 32.73 seconds

## Relational analysis of IS_A2_A1_A2_A2_B2_A2_A2

### Relational analysis result of IS_A2_A1_A2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8125182, upper bound: 17.8321712
time: 18.43 seconds

## BFS IS instance: IS_A2_A2_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -28.6388741, 5.7900629, -28.5747948, 5.8138390, -34.4527130, 34.3648567
1: -15.2485867, 11.1212101, -15.1966839, 11.1423731, -26.3909607, 26.3178940
2: -12.3546801, 10.9057779, -12.3074608, 10.9195881, -23.2742691, 23.2132378
3: -8.9990845, 15.7865448, -8.9633598, 15.8017368, -24.8008213, 24.7499046
4: -12.7350712, 13.2556934, -12.7157001, 13.2760553, -26.0111275, 25.9713936
5: -9.9532995, 18.0430126, -9.9078655, 18.0597095, -28.0130081, 27.9508781
6: -27.4566193, -2.9411917, -27.4594727, -2.9790168, -18.9039764, 19.0105267
7: -13.2607632, 17.6992607, -13.2244759, 17.7152939, -30.9760571, 30.9237366
8: -16.9920769, 15.7914133, -16.9561024, 15.8120155, -31.7624207, 31.7112236
9: -12.2407265, 13.5645618, -12.2133789, 13.5887680, -21.3780518, 21.2738571
10: -13.1181488, 24.7147732, -13.0258474, 24.7481823, -34.6874390, 34.5241585
11: -22.7171478, 12.7985020, -22.6892204, 12.8099966, -33.8535233, 33.8188133
12: -20.8423500, 15.4410667, -20.8507175, 15.4355392, -36.1134262, 36.1856346
13: -21.0448227, 11.3133488, -21.0413189, 11.3166676, -25.8451614, 25.7664948
14: -43.0592079, 3.4260755, -42.9796982, 3.4472055, -34.3975945, 34.2345848
15: -15.1410599, 9.8656139, -15.0854931, 9.8738909, -24.4793320, 24.4165649
16: -21.1467285, 13.1318607, -21.0921402, 13.1559238, -33.3615417, 33.2597961
17: -33.9255142, 27.4964542, -33.8215866, 27.5082092, -52.5204468, 52.3618774
18: -17.6544685, 7.9515357, -17.6625080, 7.9535327, -24.3208961, 24.3644543
19: -20.0829105, 2.0152903, -20.0954514, 2.0134072, -21.4641571, 21.5021515
20: -10.1480122, 10.3078318, -10.1633377, 10.2677059, -19.6555481, 19.7378006
21: -20.6855221, 7.2068892, -20.6927147, 7.1938777, -27.8794003, 27.8996048
22: -22.9044266, 9.3267584, -22.9161148, 9.3275318, -31.3075714, 31.3386536
23: -19.3507767, 4.2518353, -19.3582840, 4.2600851, -22.3059616, 22.3379898
24: -26.7353859, -1.6887312, -26.7456875, -1.7132444, -21.4032593, 21.4841423
25: -13.2916651, 9.5081472, -13.2958059, 9.4858179, -21.3087540, 21.3624725
26: -28.8995628, 8.7522783, -28.9275017, 8.7597694, -37.5939026, 37.6731339
27: -28.5694065, 0.3366165, -28.5866108, 0.3063636, -24.4020615, 24.5096931
28: -18.5110149, 6.3374529, -18.5356350, 6.3072519, -23.9252090, 23.9974747
29: -32.0640068, 5.0048265, -32.0686035, 5.0365372, -35.7328873, 35.7272263
30: -18.4732609, 8.4038792, -18.4878960, 8.3837814, -25.7008705, 25.7737236
31: -18.0074348, 8.5145721, -18.0068626, 8.4948626, -25.0653534, 25.1080990
32: -21.4044437, 4.2854228, -21.4086304, 4.2169542, -22.2569122, 22.3533478
33: -39.3173447, 1.2026644, -39.3185844, 1.0944786, -32.2071762, 32.3434830
34: -30.8209782, 2.2621870, -30.8177681, 2.1755791, -27.7850227, 27.8927078
35: -30.2939110, 2.5467582, -30.3061981, 2.4585776, -26.1962471, 26.3158913
36: -31.7277985, 0.2516983, -31.7528763, 0.1668077, -24.4600677, 24.6068802
37: -47.3454437, -6.4774294, -47.3557663, -6.5440526, -32.5398674, 32.6575127
38: -40.6347504, -2.1145535, -40.6593361, -2.1987734, -27.6590271, 27.8315220
39: -50.5516129, -5.8490515, -50.5576477, -5.9590893, -34.4068222, 34.5536690
40: -41.6932678, -3.3344193, -41.7029839, -3.4025669, -31.7341995, 31.8551254
41: -31.1547298, -4.2195339, -31.1606293, -4.2518702, -19.9541550, 20.0392780
42: -18.1180630, 2.5600057, -18.1203175, 2.5682545, -19.5020752, 19.5536880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=101, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1313
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 731

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_A2_A1_A1_A2_B2_A1

### Relational analysis result of IS_A2_A2_A1_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8125182, upper bound: 17.8775739
time: 19.76 seconds

## Relational analysis of IS_A2_A2_A1_A1_A2_B2_A2

### Relational analysis result of IS_A2_A2_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8125182, upper bound: 17.8864980
time: 14.98 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -28.7132263, 5.7999921, -28.6103821, 5.8166442, -34.5298691, 34.4103737
1: -15.2838984, 11.1304369, -15.2132225, 11.1455021, -26.4294014, 26.3436584
2: -12.3971634, 10.9159069, -12.3275175, 10.9225063, -23.3196697, 23.2434235
3: -9.0503464, 15.8094110, -8.9876518, 15.8086662, -24.8590126, 24.7970619
4: -12.7715740, 13.2724762, -12.7329473, 13.2815008, -26.0530739, 26.0054245
5: -9.9985838, 18.0585651, -9.9292564, 18.0635986, -28.0621834, 27.9878216
6: -27.4746094, -2.8985515, -27.4674263, -2.9584174, -18.9450569, 19.0509987
7: -13.3026562, 17.7102318, -13.2440624, 17.7202301, -31.0228863, 30.9542942
8: -17.0468979, 15.8089952, -16.9818592, 15.8187428, -31.8270721, 31.7709808
9: -12.3126507, 13.5930996, -12.2479334, 13.5948982, -21.4285355, 21.3381271
10: -13.1936913, 24.7458076, -13.0613546, 24.7565975, -34.7540359, 34.5908165
11: -22.7482109, 12.8672934, -22.6962681, 12.8429737, -33.9183807, 33.8881760
12: -20.8680000, 15.4580641, -20.8625774, 15.4424353, -36.1779900, 36.2236748
13: -21.1221619, 11.3454390, -21.0784931, 11.3238430, -25.8975067, 25.8370476
14: -43.1562271, 3.4411836, -43.0261040, 3.4514284, -34.4732857, 34.2973633
15: -15.1718140, 9.8831806, -15.1001711, 9.8811569, -24.5193672, 24.4545784
16: -21.2155247, 13.1484766, -21.1242828, 13.1602049, -33.4228821, 33.3081970
17: -33.9967728, 27.5129128, -33.8528366, 27.5161381, -52.6073914, 52.4143143
18: -17.6843033, 8.0165539, -17.6698112, 7.9847488, -24.3831406, 24.4118881
19: -20.1077576, 2.0755167, -20.1001835, 2.0425334, -21.5174904, 21.5513687
20: -10.1687660, 10.3702765, -10.1670408, 10.2978153, -19.7076912, 19.7986298
21: -20.7153378, 7.2771935, -20.6987400, 7.2277765, -27.9431152, 27.9759331
22: -22.9356461, 9.4157953, -22.9229660, 9.3706188, -31.3819199, 31.4320908
23: -19.3759346, 4.3059893, -19.3632832, 4.2862177, -22.3571625, 22.3891487
24: -26.7702370, -1.6068149, -26.7537785, -1.6736474, -21.4775772, 21.5455132
25: -13.3134899, 9.5693846, -13.3009253, 9.5154877, -21.3604317, 21.4280548
26: -28.9312859, 8.8448372, -28.9328918, 8.8043499, -37.6684265, 37.7523117
27: -28.6036930, 0.4247756, -28.5925579, 0.3489017, -24.4777260, 24.5716057
28: -18.5336304, 6.4009438, -18.5387154, 6.3378716, -23.9779663, 24.0547333
29: -32.1041451, 5.1087265, -32.0788116, 5.0866461, -35.8234024, 35.8391113
30: -18.5019760, 8.4735107, -18.4942036, 8.4173336, -25.7637749, 25.8441353
31: -18.0388870, 8.5712624, -18.0155735, 8.5222406, -25.1248131, 25.1610603
32: -21.4233856, 4.3142319, -21.4158039, 4.2304225, -22.3012733, 22.3956947
33: -39.3379250, 1.2333755, -39.3274879, 1.1088243, -32.2439499, 32.3855820
34: -30.8345432, 2.2931485, -30.8242874, 2.1897411, -27.8085861, 27.9352608
35: -30.3119907, 2.5695214, -30.3142509, 2.4689465, -26.2264023, 26.3498116
36: -31.7451096, 0.3028276, -31.7602730, 0.1916592, -24.5037422, 24.6598816
37: -47.3769264, -6.4200020, -47.3682404, -6.5166273, -32.6047134, 32.7139435
38: -40.6580658, -2.0592108, -40.6676331, -2.1729007, -27.7009354, 27.8853092
39: -50.5785904, -5.8309784, -50.5696182, -5.9512115, -34.4436493, 34.5900879
40: -41.7121086, -3.3031082, -41.7120743, -3.3880343, -31.7648849, 31.8892975
41: -31.1762638, -4.1692314, -31.1693554, -4.2276096, -20.0053043, 20.0851192
42: -18.1329231, 2.5946116, -18.1254902, 2.5847125, -19.5426044, 19.5949440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=101, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=212, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1314
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 731

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_A2_A1_A2_B2_A2_A1

### Relational analysis result of IS_A2_A2_A1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8338529, upper bound: 17.8830306
time: 25.92 seconds

## Relational analysis of IS_A2_A2_A1_A2_B2_A2_A2

### Relational analysis result of IS_A2_A2_A1_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8338529, upper bound: 17.8468849
time: 33.19 seconds

## BFS IS instance: IS_A2_A2_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -28.6667595, 5.8143406, -28.5869370, 5.8151278, -34.4818878, 34.4012756
1: -15.2704239, 11.1444283, -15.2070370, 11.1438742, -26.4142990, 26.3514652
2: -12.3668270, 10.9195404, -12.3129845, 10.9210854, -23.2879124, 23.2325249
3: -9.0090647, 15.7964687, -8.9677725, 15.8029699, -24.8120346, 24.7642403
4: -12.7511444, 13.2735901, -12.7233143, 13.2772989, -26.0284424, 25.9969044
5: -9.9624920, 18.0523834, -9.9118376, 18.0604248, -28.0229168, 27.9642220
6: -27.4640884, -2.9351478, -27.4624271, -2.9776592, -18.9363136, 19.0106430
7: -13.2790165, 17.7170715, -13.2326975, 17.7165794, -30.9955959, 30.9497681
8: -17.0118275, 15.8146400, -16.9655743, 15.8140306, -31.7711182, 31.7446365
9: -12.2537823, 13.5730705, -12.2190351, 13.5894775, -21.3854980, 21.2904854
10: -13.1453142, 24.7379913, -13.0377951, 24.7499886, -34.7149277, 34.5589981
11: -22.7279739, 12.8030777, -22.6930809, 12.8112621, -33.8645058, 33.8320084
12: -20.8504181, 15.4518862, -20.8522530, 15.4374084, -36.1216049, 36.1980476
13: -21.0581074, 11.3270664, -21.0430756, 11.3225307, -25.8579750, 25.7758331
14: -43.0825233, 3.4444866, -42.9883308, 3.4490190, -34.4131737, 34.2617264
15: -15.1521187, 9.8721008, -15.0896511, 9.8757610, -24.4908714, 24.4268990
16: -21.1756592, 13.1517448, -21.1051788, 13.1571770, -33.3802643, 33.2925568
17: -33.9420891, 27.5074902, -33.8263626, 27.5096302, -52.5379791, 52.3832932
18: -17.6670094, 7.9629564, -17.6669998, 7.9557328, -24.3365250, 24.3810654
19: -20.0966759, 2.0279670, -20.0975857, 2.0190458, -21.4836426, 21.5148468
20: -10.1554680, 10.3133917, -10.1651630, 10.2698689, -19.6748886, 19.7435303
21: -20.6939621, 7.2096119, -20.6950111, 7.1950636, -27.8890266, 27.9046230
22: -22.9136600, 9.3334198, -22.9184532, 9.3302212, -31.3190155, 31.3467255
23: -19.3635921, 4.2651868, -19.3594551, 4.2662163, -22.3246765, 22.3461990
24: -26.7363853, -1.6861405, -26.7449322, -1.7114220, -21.4085617, 21.4903183
25: -13.2947598, 9.5141172, -13.2957573, 9.4884071, -21.3151474, 21.3685493
26: -28.9177418, 8.7679253, -28.9297867, 8.7666273, -37.6211853, 37.6907425
27: -28.5746498, 0.3405566, -28.5879517, 0.3075385, -24.4204102, 24.5128708
28: -18.5258369, 6.3519697, -18.5367393, 6.3141851, -23.9470062, 24.0116882
29: -32.0745163, 5.0072460, -32.0715256, 5.0371571, -35.7442703, 35.7317429
30: -18.4803772, 8.4071217, -18.4894543, 8.3857594, -25.7115250, 25.7774658
31: -18.0156441, 8.5245686, -18.0091171, 8.4987640, -25.0775375, 25.1201515
32: -21.4120827, 4.2900023, -21.4108105, 4.2182021, -22.2691612, 22.3599319
33: -39.3214798, 1.2110615, -39.3193970, 1.0980926, -32.2145691, 32.3523407
34: -30.8281460, 2.2701874, -30.8194313, 2.1789517, -27.7950020, 27.8998795
35: -30.3110256, 2.5626426, -30.3073044, 2.4660420, -26.2210732, 26.3236618
36: -31.7561512, 0.2768393, -31.7550850, 0.1789167, -24.4994125, 24.6233482
37: -47.3545113, -6.4722614, -47.3577461, -6.5418911, -32.5530739, 32.6641502
38: -40.6565781, -2.1008105, -40.6616478, -2.1929550, -27.6891479, 27.8382912
39: -50.5563660, -5.8417311, -50.5584030, -5.9559364, -34.4148941, 34.5580444
40: -41.7071152, -3.3244801, -41.7083969, -3.4007325, -31.7474403, 31.8597260
41: -31.1634121, -4.2126489, -31.1628304, -4.2491837, -19.9705639, 20.0501881
42: -18.1296883, 2.5681701, -18.1252937, 2.5693951, -19.5161743, 19.5557594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=101, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1314
type: B, layer: 1, pos: 1313
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 731

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_A2_A2_A1_A2_B2_A1

### Relational analysis result of IS_A2_A2_A2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8266918, upper bound: 17.8786762
time: 24.12 seconds

## Relational analysis of IS_A2_A2_A2_A1_A2_B2_A2

### Relational analysis result of IS_A2_A2_A2_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8266918, upper bound: 17.8786762
time: 26.41 seconds

## BFS IS instance: IS_A2_A2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -28.6817589, 5.8122072, -28.6224995, 5.8170276, -34.4987869, 34.4347076
1: -15.2626190, 11.1424160, -15.2234612, 11.1461020, -26.4087219, 26.3658772
2: -12.3802500, 10.9227028, -12.3330183, 10.9235878, -23.3038368, 23.2557220
3: -9.0343113, 15.8095684, -8.9920635, 15.8094749, -24.8437862, 24.8016319
4: -12.7655029, 13.2823610, -12.7405481, 13.2823648, -26.0478668, 26.0229092
5: -9.9734440, 18.0600128, -9.9332275, 18.0636883, -28.0371323, 27.9932404
6: -27.4687786, -2.9156518, -27.4697800, -2.9570355, -18.9648323, 19.0295868
7: -13.2784538, 17.7161846, -13.2521410, 17.7206306, -30.9990845, 30.9683266
8: -17.0323849, 15.8144188, -16.9913216, 15.8199682, -31.8001556, 31.7868881
9: -12.2994919, 13.5944233, -12.2536097, 13.5952435, -21.4099998, 21.3475685
10: -13.1576176, 24.7542648, -13.0732985, 24.7573853, -34.7178192, 34.6119995
11: -22.7252808, 12.8600063, -22.7001438, 12.8438482, -33.8936234, 33.8890839
12: -20.8662605, 15.4596319, -20.8648338, 15.4444256, -36.1740417, 36.2233276
13: -21.1272068, 11.3508015, -21.0815735, 11.3298626, -25.8922501, 25.8410988
14: -43.1214066, 3.4497185, -43.0348892, 3.4525247, -34.4307098, 34.3150215
15: -15.1621189, 9.8812637, -15.1043549, 9.8828468, -24.5093307, 24.4574051
16: -21.1865158, 13.1540632, -21.1373634, 13.1603031, -33.3821106, 33.3270874
17: -33.9138603, 27.4989185, -33.8576736, 27.5154839, -52.5231628, 52.4112320
18: -17.6891804, 8.0221682, -17.6746120, 7.9879341, -24.3875732, 24.4206676
19: -20.1148663, 2.0823517, -20.1023331, 2.0487700, -21.5316162, 21.5586243
20: -10.1655912, 10.3414335, -10.1683197, 10.2999802, -19.7171173, 19.7693672
21: -20.7125874, 7.2726922, -20.7010136, 7.2289357, -27.9415226, 27.9737053
22: -22.9279823, 9.4025373, -22.9247169, 9.3732958, -31.3773422, 31.4183578
23: -19.3814545, 4.3173714, -19.3645554, 4.2943192, -22.3720779, 22.3875198
24: -26.7581539, -1.6243606, -26.7530899, -1.6718636, -21.4708900, 21.5310936
25: -13.3108549, 9.5597591, -13.3010139, 9.5180779, -21.3630486, 21.4129448
26: -28.9405251, 8.8408680, -28.9351540, 8.8112259, -37.6889572, 37.7418976
27: -28.5962830, 0.3965697, -28.5934563, 0.3500848, -24.4839821, 24.5419617
28: -18.5415955, 6.3867359, -18.5396080, 6.3447719, -23.9930420, 24.0399704
29: -32.0944595, 5.1063232, -32.0816803, 5.0872574, -35.8173828, 35.8343124
30: -18.4993401, 8.4687920, -18.4966774, 8.4194527, -25.7681656, 25.8274002
31: -18.0343876, 8.5666637, -18.0177116, 8.5260601, -25.1248550, 25.1586380
32: -21.4121132, 4.2706671, -21.4167271, 4.2318268, -22.2967033, 22.3511925
33: -39.3247757, 1.1781440, -39.3271255, 1.1123734, -32.2350159, 32.3296013
34: -30.8225250, 2.2418089, -30.8248005, 2.1931005, -27.8006668, 27.8818779
35: -30.3095093, 2.5226884, -30.3141766, 2.4763689, -26.2323112, 26.2935524
36: -31.7535572, 0.2559171, -31.7608566, 0.2037640, -24.5250969, 24.6019554
37: -47.3696251, -6.4562459, -47.3696251, -6.5145330, -32.6022530, 32.6785355
38: -40.6595268, -2.1198883, -40.6682320, -2.1669979, -27.7119827, 27.8127594
39: -50.5628128, -5.8930817, -50.5688667, -5.9479513, -34.4323120, 34.5236320
40: -41.7109833, -3.3388319, -41.7166519, -3.3862543, -31.7633820, 31.8469086
41: -31.1713448, -4.1891599, -31.1708908, -4.2248635, -20.0086632, 20.0682907
42: -18.1393356, 2.5997105, -18.1303596, 2.5875258, -19.5520191, 19.5900993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=101, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_A2_A2_A2_B2_A1_A1

### Relational analysis result of IS_A2_A2_A2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8480135, upper bound: 17.8697671
time: 21.28 seconds

## Relational analysis of IS_A2_A2_A2_A2_B2_A1_A2

### Relational analysis result of IS_A2_A2_A2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8480135, upper bound: 17.8336578
time: 19.06 seconds

## BFS IS instance: IS_A2_A2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -28.7411003, 5.8242702, -28.6225471, 5.8179379, -34.5590363, 34.4468155
1: -15.3057585, 11.1536369, -15.2235403, 11.1469564, -26.4527149, 26.3771782
2: -12.4093351, 10.9296675, -12.3330631, 10.9240055, -23.3333397, 23.2627296
3: -9.0603342, 15.8193445, -8.9920826, 15.8098888, -24.8702240, 24.8114281
4: -12.7876816, 13.2903643, -12.7405415, 13.2827168, -26.0703983, 26.0309067
5: -10.0077991, 18.0679455, -9.9332466, 18.0643215, -28.0721207, 28.0011921
6: -27.4820728, -2.8925500, -27.4703484, -2.9570689, -18.9773788, 19.0511169
7: -13.3209276, 17.7280750, -13.2522421, 17.7215176, -31.0424461, 30.9803162
8: -17.0666656, 15.8321981, -16.9913597, 15.8207760, -31.8357468, 31.8043900
9: -12.3257160, 13.6015930, -12.2536049, 13.5956125, -21.4359741, 21.3547440
10: -13.2208366, 24.7690163, -13.0733194, 24.7584038, -34.7815247, 34.6256561
11: -22.7590485, 12.8718567, -22.7001076, 12.8442421, -33.9293365, 33.9013596
12: -20.8760471, 15.4688597, -20.8641052, 15.4442921, -36.1861458, 36.2360344
13: -21.1354103, 11.3591385, -21.0802383, 11.3297119, -25.9103165, 25.8463860
14: -43.1795769, 3.4596438, -43.0347862, 3.4532461, -34.4888649, 34.3245049
15: -15.1828756, 9.8896494, -15.1043396, 9.8830261, -24.5309258, 24.4648972
16: -21.2444344, 13.1683311, -21.1373520, 13.1614370, -33.4415970, 33.3409424
17: -34.0133324, 27.5239487, -33.8576279, 27.5174942, -52.6248779, 52.4357300
18: -17.6968403, 8.0279465, -17.6742706, 7.9869614, -24.3987541, 24.4284916
19: -20.1215000, 2.0881667, -20.1023178, 2.0481706, -21.5369873, 21.5640564
20: -10.1762323, 10.3758287, -10.1688652, 10.2999840, -19.7270164, 19.8043671
21: -20.7238140, 7.2799110, -20.7010536, 7.2289505, -27.9527645, 27.9809647
22: -22.9449196, 9.4224539, -22.9253101, 9.3733263, -31.3933868, 31.4402008
23: -19.3887501, 4.3193426, -19.3644466, 4.2923336, -22.3758621, 22.3973541
24: -26.7712345, -1.6042590, -26.7530098, -1.6718340, -21.4828873, 21.5516930
25: -13.3165894, 9.5753851, -13.3008947, 9.5180712, -21.3668671, 21.4341507
26: -28.9494705, 8.8605013, -28.9352264, 8.8112202, -37.6956940, 37.7699051
27: -28.6089420, 0.4287314, -28.5938988, 0.3500438, -24.4960632, 24.5747986
28: -18.5484428, 6.4154253, -18.5398064, 6.3447819, -23.9997787, 24.0689812
29: -32.1146507, 5.1111231, -32.0817299, 5.0873203, -35.8347549, 35.8436127
30: -18.5091209, 8.4767685, -18.4957504, 8.4192886, -25.7744217, 25.8479004
31: -18.0470886, 8.5812607, -18.0178242, 8.5260983, -25.1370010, 25.1731186
32: -21.4310493, 4.3188391, -21.4179287, 4.2316580, -22.3135185, 22.4022408
33: -39.3420296, 1.2418127, -39.3282967, 1.1124473, -32.2513962, 32.3944283
34: -30.8417282, 2.3011484, -30.8259544, 2.1931400, -27.8185616, 27.9424515
35: -30.3290977, 2.5853682, -30.3153992, 2.4764142, -26.2512054, 26.3576012
36: -31.7734756, 0.3279684, -31.7624702, 0.2037852, -24.5430946, 24.6763496
37: -47.3859940, -6.4148369, -47.3702583, -6.5144572, -32.6178741, 32.7205734
38: -40.6798477, -2.0454617, -40.6699181, -2.1670914, -27.7310486, 27.8920593
39: -50.5833893, -5.8236389, -50.5703201, -5.9480114, -34.4517288, 34.5944595
40: -41.7259483, -3.2931786, -41.7175140, -3.3862352, -31.7781563, 31.8939056
41: -31.1849537, -4.1623664, -31.1715508, -4.2249413, -20.0217209, 20.0960159
42: -18.1445427, 2.6027694, -18.1304665, 2.5858603, -19.5567188, 19.5970268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=101, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=212, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1333
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1410
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 731

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_A2_A2_A2_B2_A2_A1

### Relational analysis result of IS_A2_A2_A2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8125182, upper bound: 17.8830311
time: 30.64 seconds

## Relational analysis of IS_A2_A2_A2_A2_B2_A2_A2

### Relational analysis result of IS_A2_A2_A2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8125182, upper bound: 17.8480134
time: 19.70 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 52.61 seconds
IS_A2_A1_A2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 52.61
Output dim: 10, lower bound: -17.8480135, upper bound: 17.8598913
IS_A2_A1_A2_A2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 52.61
Output dim: 10, lower bound: -17.8480135, upper bound: 17.8238006
IS_A2_A1_A2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 52.61
Output dim: 10, lower bound: -17.8125182, upper bound: 17.8742360
IS_A2_A1_A2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 52.61
Output dim: 10, lower bound: -17.8125182, upper bound: 17.8321712
IS_A2_A2_A1_A1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 52.61
Output dim: 10, lower bound: -17.8125182, upper bound: 17.8775739
IS_A2_A2_A1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 52.61
Output dim: 10, lower bound: -17.8125182, upper bound: 17.8864980
IS_A2_A2_A1_A2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 52.61
Output dim: 10, lower bound: -17.8338529, upper bound: 17.8830306
IS_A2_A2_A1_A2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 52.61
Output dim: 10, lower bound: -17.8338529, upper bound: 17.8468849
IS_A2_A2_A2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 52.61
Output dim: 10, lower bound: -17.8266918, upper bound: 17.8786762
IS_A2_A2_A2_A1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 52.61
Output dim: 10, lower bound: -17.8266918, upper bound: 17.8786762
IS_A2_A2_A2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 52.61
Output dim: 10, lower bound: -17.8480135, upper bound: 17.8697671
IS_A2_A2_A2_A2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 52.61
Output dim: 10, lower bound: -17.8480135, upper bound: 17.8336578
IS_A2_A2_A2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 52.61
Output dim: 10, lower bound: -17.8125182, upper bound: 17.8830311
IS_A2_A2_A2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 52.61
Output dim: 10, lower bound: -17.8125182, upper bound: 17.8480134

## BFS IS instance: IS_A2_A2_A1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -28.6365814, 5.7898598, -28.5747948, 5.8138390, -34.4504204, 34.3646545
1: -15.2474003, 11.1210880, -15.1966839, 11.1423731, -26.3897743, 26.3177719
2: -12.3534899, 10.9056644, -12.3074608, 10.9195881, -23.2730789, 23.2131252
3: -8.9978876, 15.7861662, -8.9633598, 15.8017368, -24.7996254, 24.7495270
4: -12.7340202, 13.2554665, -12.7157001, 13.2760553, -26.0100746, 25.9711666
5: -9.9520741, 18.0428562, -9.9078655, 18.0597095, -28.0117836, 27.9507217
6: -27.4562778, -2.9426212, -27.4594727, -2.9790168, -18.9036331, 18.9698868
7: -13.2595329, 17.6990356, -13.2244759, 17.7152939, -30.9748268, 30.9235115
8: -16.9905357, 15.7909994, -16.9561024, 15.8120155, -31.7608566, 31.7111130
9: -12.2390232, 13.5642738, -12.2133789, 13.5887680, -21.3330917, 21.2735634
10: -13.1162672, 24.7144928, -13.0258474, 24.7481823, -34.6530914, 34.5238800
11: -22.7167549, 12.7977686, -22.6892204, 12.8099966, -33.8531342, 33.8098068
12: -20.8419380, 15.4397392, -20.8507175, 15.4355392, -36.1419907, 36.1844940
13: -21.0432739, 11.3130112, -21.0413189, 11.3166676, -25.8038940, 25.7660751
14: -43.0568390, 3.4259768, -42.9796982, 3.4472055, -34.3458900, 34.2344398
15: -15.1398268, 9.8652992, -15.0854931, 9.8738909, -24.4721222, 24.4162560
16: -21.1450844, 13.1316500, -21.0921402, 13.1559238, -33.3418732, 33.2595940
17: -33.9238548, 27.4961052, -33.8215866, 27.5082092, -52.4844666, 52.3605804
18: -17.6541290, 7.9502773, -17.6625080, 7.9535327, -24.3205643, 24.3255730
19: -20.0826473, 2.0140104, -20.0954514, 2.0134072, -21.4639130, 21.4811172
20: -10.1478004, 10.3063488, -10.1633377, 10.2677059, -19.6553650, 19.7208138
21: -20.6851749, 7.2053471, -20.6927147, 7.1938777, -27.8790531, 27.8980618
22: -22.9040451, 9.3251133, -22.9161148, 9.3275318, -31.3071747, 31.3328552
23: -19.3504372, 4.2508178, -19.3582840, 4.2600851, -22.3056564, 22.3305511
24: -26.7348423, -1.6904616, -26.7456875, -1.7132444, -21.4027023, 21.4449921
25: -13.2913847, 9.5068054, -13.2958059, 9.4858179, -21.3084869, 21.3584671
26: -28.8992672, 8.7503681, -28.9275017, 8.7597694, -37.5936127, 37.6507034
27: -28.5691490, 0.3347812, -28.5866108, 0.3063636, -24.4018021, 24.4647064
28: -18.5108624, 6.3360434, -18.5356350, 6.3072519, -23.9250641, 23.9844131
29: -32.0634041, 5.0031013, -32.0686035, 5.0365372, -35.7322693, 35.7216034
30: -18.4728737, 8.4024563, -18.4878960, 8.3837814, -25.7005272, 25.7610855
31: -18.0068893, 8.5131550, -18.0068626, 8.4948626, -25.0647964, 25.0890789
32: -21.4041672, 4.2843499, -21.4086304, 4.2169542, -22.2566681, 22.3503494
33: -39.3170357, 1.2013950, -39.3185844, 1.0944786, -32.2068634, 32.3254471
34: -30.8207550, 2.2610970, -30.8177681, 2.1755791, -27.7847786, 27.8682365
35: -30.2935638, 2.5457869, -30.3061981, 2.4585776, -26.1958961, 26.2988472
36: -31.7275047, 0.2501388, -31.7528763, 0.1668077, -24.4598312, 24.5756950
37: -47.3448715, -6.4787903, -47.3557663, -6.5440526, -32.5392570, 32.6245880
38: -40.6344719, -2.1162362, -40.6593361, -2.1987734, -27.6587296, 27.7733040
39: -50.5512047, -5.8500433, -50.5576477, -5.9590893, -34.4064255, 34.5326042
40: -41.6930313, -3.3354254, -41.7029839, -3.4025669, -31.7338943, 31.8278961
41: -31.1543522, -4.2204099, -31.1606293, -4.2518702, -19.9537964, 19.9978409
42: -18.1178589, 2.5596538, -18.1203175, 2.5682545, -19.5074654, 19.5527496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=100, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1426
type: A, layer: 1, pos: 1426
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1333
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1410
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1329
type: B, layer: 1, pos: 1329
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1394
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1394
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1314
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 731

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1699

## Relational analysis of IS_A2_A2_A1_A1_A2_B2_A2_A1

### Relational analysis result of IS_A2_A2_A1_A1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8102918, upper bound: 17.8242257
time: 21.71 seconds

## Relational analysis of IS_A2_A2_A1_A1_A2_B2_A2_A2

### Relational analysis result of IS_A2_A2_A1_A1_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8102918, upper bound: 17.8398460
time: 16.16 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 40.14 seconds
IS_A2_A2_A1_A1_A2_B2_A2_A1, status: Status.VERIFIED, split count: 8, time: 40.14
Output dim: 10, lower bound: -17.8102918, upper bound: 17.8242257
IS_A2_A2_A1_A1_A2_B2_A2_A2, status: Status.VERIFIED, split count: 8, time: 40.14
Output dim: 10, lower bound: -17.8102918, upper bound: 17.8398460

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 32.56 + 1453.80 = 1486.36 seconds
