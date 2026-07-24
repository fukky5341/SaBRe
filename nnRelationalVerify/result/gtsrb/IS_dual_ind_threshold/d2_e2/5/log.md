## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 5)
Time budget: 3600 seconds
Split limit: 100
Threshold: 40.2367993236


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=571, inp2_unstable=571, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-60.4618378, 24.9711685, -60.4618378, 24.9711685, -85.4330063, 85.4330063)
1: (-31.5622578, 29.6370068, -31.5622578, 29.6370068, -61.1992607, 61.1992645)
2: (-27.9061604, 26.5470467, -27.9061604, 26.5470467, -54.4532089, 54.4532051)
3: (-31.8409405, 32.9238777, -31.8409405, 32.9238777, -64.7648163, 64.7648163)
4: (-33.5803528, 33.3845100, -33.5803528, 33.3845100, -66.9648590, 66.9648590)
5: (-30.2060833, 34.3865662, -30.2060833, 34.3865662, -64.5926514, 64.5926514)
6: (-42.2196693, 28.9583435, -42.2196693, 28.9583435, -71.1780090, 71.1780090)
7: (-35.3623466, 37.3481178, -35.3623466, 37.3481178, -72.7104568, 72.7104645)
8: (-37.5573502, 41.5086517, -37.5573502, 41.5086517, -79.0660019, 79.0660019)
9: (-33.8015938, 31.1720695, -33.8015938, 31.1720695, -64.9736633, 64.9736633)
10: (-47.6147919, 44.8453522, -47.6147919, 44.8453522, -92.4601440, 92.4601440)
11: (-45.6412048, 35.5001945, -45.6412048, 35.5001945, -81.1413879, 81.1413956)
12: (-46.8553925, 35.8280334, -46.8553925, 35.8280334, -82.6834259, 82.6834259)
13: (-48.2150345, 41.6089211, -48.2150345, 41.6089211, -89.8239441, 89.8239517)
14: (-75.6748352, 32.2548332, -75.6748352, 32.2548332, -107.9296722, 107.9296646)
15: (-39.2797165, 27.5555019, -39.2797165, 27.5555019, -66.8352203, 66.8352203)
16: (-48.4652214, 36.7890320, -48.4652214, 36.7890320, -85.2542419, 85.2542496)
17: (-75.6703949, 50.9458008, -75.6703949, 50.9458008, -126.6161957, 126.6161880)
18: (-41.8936539, 33.3293495, -41.8936539, 33.3293495, -75.2230072, 75.2230072)
19: (-33.7314529, 19.6711330, -33.7314529, 19.6711330, -53.4025841, 53.4025879)
20: (-30.1084747, 23.0905399, -30.1084747, 23.0905399, -53.1990128, 53.1990128)
21: (-44.1090546, 25.1586304, -44.1090546, 25.1586304, -69.2676849, 69.2676849)
22: (-50.1980705, 24.6493435, -50.1980705, 24.6493435, -74.8474121, 74.8474121)
23: (-35.4210587, 25.4334450, -35.4210587, 25.4334450, -60.8544998, 60.8545036)
24: (-43.5227928, 28.3837967, -43.5227928, 28.3837967, -71.9065857, 71.9065857)
25: (-36.2998810, 29.4436989, -36.2998810, 29.4436989, -65.7435760, 65.7435760)
26: (-48.6222916, 36.2146683, -48.6222916, 36.2146683, -84.8369598, 84.8369598)
27: (-44.2783661, 30.0313797, -44.2783661, 30.0313797, -74.3097382, 74.3097458)
28: (-33.9707375, 28.5353203, -33.9707375, 28.5353203, -62.5060577, 62.5060577)
29: (-56.0578918, 26.7704124, -56.0578918, 26.7704124, -82.8283081, 82.8283081)
30: (-40.6937065, 34.5762024, -40.6937065, 34.5762024, -75.2699127, 75.2699127)
31: (-43.6006355, 27.4441547, -43.6006355, 27.4441547, -71.0447922, 71.0447922)
32: (-45.7585983, 25.4946594, -45.7585983, 25.4946594, -71.2532578, 71.2532578)
33: (-64.4699249, 32.4027214, -64.4699249, 32.4027214, -96.8726425, 96.8726501)
34: (-53.7206001, 23.8197269, -53.7206001, 23.8197269, -77.5403290, 77.5403290)
35: (-55.1624718, 25.6209793, -55.1624718, 25.6209793, -80.7834473, 80.7834473)
36: (-52.9996796, 27.5723953, -52.9996796, 27.5723953, -80.5720673, 80.5720673)
37: (-73.7018433, 26.4361076, -73.7018433, 26.4361076, -100.1379547, 100.1379547)
38: (-64.1630859, 32.1271362, -64.1630859, 32.1271362, -96.2902222, 96.2902222)
39: (-76.1108093, 31.5794048, -76.1108093, 31.5794048, -107.6902161, 107.6902161)
40: (-60.8784599, 27.9714222, -60.8784599, 27.9714222, -88.8498840, 88.8498840)
41: (-44.7025604, 25.6049080, -44.7025604, 25.6049080, -70.3074646, 70.3074646)
42: (-32.4942627, 23.6886921, -32.4942627, 23.6886921, -56.1829529, 56.1829529)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.42 + 123.17 = 125.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -40.2770764, upper bound: 40.2770764

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1569

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -40.1910291, upper bound: 40.2340530
time: 93.16 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -40.1910291, upper bound: 40.2679349
time: 85.18 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 178.45 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 178.45
Output dim: 4, lower bound: -40.1910291, upper bound: 40.2340530
IS_A2, status: Status.UNKNOWN, split count: 1, time: 178.45
Output dim: 4, lower bound: -40.1910291, upper bound: 40.2679349

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -60.4556122, 24.9667034, -60.4618378, 24.9711685, -85.4267731, 85.4285431
1: -31.5587482, 29.6322403, -31.5622578, 29.6370068, -61.1957550, 61.1944962
2: -27.9020786, 26.5451794, -27.9061604, 26.5470467, -54.4491196, 54.4513397
3: -31.8347282, 32.9198380, -31.8409405, 32.9238777, -64.7586060, 64.7607803
4: -33.5688477, 33.3822021, -33.5803528, 33.3845100, -66.9533539, 66.9625549
5: -30.1998024, 34.3840065, -30.2060833, 34.3865662, -64.5863647, 64.5900879
6: -42.2168274, 28.9444275, -42.2196693, 28.9583435, -71.1751709, 71.1640930
7: -35.3593826, 37.3373718, -35.3623466, 37.3481178, -72.7074966, 72.6997223
8: -37.5506783, 41.5060768, -37.5573502, 41.5086517, -79.0593185, 79.0634308
9: -33.7989578, 31.1671066, -33.8015938, 31.1720695, -64.9710236, 64.9686890
10: -47.6112900, 44.8319855, -47.6147919, 44.8453522, -92.4566422, 92.4467773
11: -45.6387024, 35.4922905, -45.6412048, 35.5001945, -81.1389008, 81.1334991
12: -46.8527298, 35.8163147, -46.8553925, 35.8280334, -82.6807556, 82.6717072
13: -48.2105713, 41.6038513, -48.2150345, 41.6089211, -89.8194885, 89.8188858
14: -75.6693802, 32.2464714, -75.6748352, 32.2548332, -107.9242096, 107.9213104
15: -39.2572021, 27.5532188, -39.2797165, 27.5555019, -66.8126984, 66.8329315
16: -48.4606857, 36.7793808, -48.4652214, 36.7890320, -85.2496948, 85.2445984
17: -75.6664581, 50.9333000, -75.6703949, 50.9458008, -126.6122589, 126.6036987
18: -41.8854027, 33.3230095, -41.8936539, 33.3293495, -75.2147522, 75.2166595
19: -33.7271118, 19.6671696, -33.7314529, 19.6711330, -53.3982430, 53.3986206
20: -30.1045094, 23.0875511, -30.1084747, 23.0905399, -53.1950455, 53.1960258
21: -44.1062622, 25.1535645, -44.1090546, 25.1586304, -69.2648926, 69.2626190
22: -50.1796379, 24.6459999, -50.1980705, 24.6493435, -74.8289795, 74.8440704
23: -35.4187698, 25.4268761, -35.4210587, 25.4334450, -60.8522110, 60.8479309
24: -43.5164909, 28.3820114, -43.5227928, 28.3837967, -71.9002838, 71.9048004
25: -36.2940712, 29.4400368, -36.2998810, 29.4436989, -65.7377625, 65.7399139
26: -48.6122093, 36.2075996, -48.6222916, 36.2146683, -84.8268738, 84.8298798
27: -44.2714500, 30.0286331, -44.2783661, 30.0313797, -74.3028259, 74.3069992
28: -33.9674454, 28.5326309, -33.9707375, 28.5353203, -62.5027618, 62.5033684
29: -56.0483208, 26.7668076, -56.0578918, 26.7704124, -82.8187332, 82.8246994
30: -40.6908493, 34.5612411, -40.6937065, 34.5762024, -75.2670517, 75.2549438
31: -43.5969238, 27.4388466, -43.6006355, 27.4441547, -71.0410767, 71.0394821
32: -45.7560997, 25.4878426, -45.7585983, 25.4946594, -71.2507629, 71.2464447
33: -64.4620056, 32.4007263, -64.4699249, 32.4027214, -96.8647308, 96.8706512
34: -53.7156601, 23.8152504, -53.7206001, 23.8197269, -77.5353851, 77.5358505
35: -55.1549835, 25.6195412, -55.1624718, 25.6209793, -80.7759628, 80.7820129
36: -52.9943619, 27.5708084, -52.9996796, 27.5723953, -80.5667572, 80.5704803
37: -73.6920776, 26.4342976, -73.7018433, 26.4361076, -100.1281891, 100.1361389
38: -64.1554718, 32.1214142, -64.1630859, 32.1271362, -96.2826080, 96.2845001
39: -76.0981598, 31.5777340, -76.1108093, 31.5794048, -107.6775589, 107.6885376
40: -60.8722305, 27.9689827, -60.8784599, 27.9714222, -88.8436508, 88.8474426
41: -44.6994591, 25.5963039, -44.7025604, 25.6049080, -70.3043671, 70.2988663
42: -32.4920807, 23.6799088, -32.4942627, 23.6886921, -56.1807709, 56.1741676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=117, inp2_unstable=118, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=571, inp2_unstable=571, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1569

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -40.2340530, upper bound: 40.1910291
time: 130.75 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -40.2340530, upper bound: 40.2679349
time: 89.97 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 222.76 seconds
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 222.76
Output dim: 4, lower bound: -40.2340530, upper bound: 40.1910291
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 222.76
Output dim: 4, lower bound: -40.2340530, upper bound: 40.2679349

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -60.4556122, 24.9667034, -60.4556122, 24.9667034, -85.4223099, 85.4223099
1: -31.5587482, 29.6322403, -31.5587482, 29.6322403, -61.1909866, 61.1909866
2: -27.9020786, 26.5451794, -27.9020786, 26.5451794, -54.4472580, 54.4472542
3: -31.8347282, 32.9198380, -31.8347282, 32.9198380, -64.7545624, 64.7545624
4: -33.5688477, 33.3822021, -33.5688477, 33.3822021, -66.9510498, 66.9510498
5: -30.1998024, 34.3840065, -30.1998024, 34.3840065, -64.5838089, 64.5838013
6: -42.2168274, 28.9444275, -42.2168274, 28.9444275, -71.1612549, 71.1612473
7: -35.3593826, 37.3373718, -35.3593826, 37.3373718, -72.6967545, 72.6967545
8: -37.5506783, 41.5060768, -37.5506783, 41.5060768, -79.0567474, 79.0567551
9: -33.7989578, 31.1671066, -33.7989578, 31.1671066, -64.9660645, 64.9660645
10: -47.6112900, 44.8319855, -47.6112900, 44.8319855, -92.4432678, 92.4432755
11: -45.6387024, 35.4922905, -45.6387024, 35.4922905, -81.1309967, 81.1309967
12: -46.8527298, 35.8163147, -46.8527298, 35.8163147, -82.6690445, 82.6690369
13: -48.2105713, 41.6038513, -48.2105713, 41.6038513, -89.8144073, 89.8144226
14: -75.6693802, 32.2464714, -75.6693802, 32.2464714, -107.9158478, 107.9158478
15: -39.2572021, 27.5532188, -39.2572021, 27.5532188, -66.8104095, 66.8104172
16: -48.4606857, 36.7793808, -48.4606857, 36.7793808, -85.2400665, 85.2400665
17: -75.6664581, 50.9333000, -75.6664581, 50.9333000, -126.5997467, 126.5997620
18: -41.8854027, 33.3230095, -41.8854027, 33.3230095, -75.2084122, 75.2084122
19: -33.7271118, 19.6671696, -33.7271118, 19.6671696, -53.3942795, 53.3942795
20: -30.1045094, 23.0875511, -30.1045094, 23.0875511, -53.1920586, 53.1920624
21: -44.1062622, 25.1535645, -44.1062622, 25.1535645, -69.2598267, 69.2598267
22: -50.1796379, 24.6459999, -50.1796379, 24.6459999, -74.8256226, 74.8256378
23: -35.4187698, 25.4268761, -35.4187698, 25.4268761, -60.8456459, 60.8456459
24: -43.5164909, 28.3820114, -43.5164909, 28.3820114, -71.8984909, 71.8984985
25: -36.2940712, 29.4400368, -36.2940712, 29.4400368, -65.7341080, 65.7341080
26: -48.6122093, 36.2075996, -48.6122093, 36.2075996, -84.8198090, 84.8198090
27: -44.2714500, 30.0286331, -44.2714500, 30.0286331, -74.3000793, 74.3000793
28: -33.9674454, 28.5326309, -33.9674454, 28.5326309, -62.5000763, 62.5000763
29: -56.0483208, 26.7668076, -56.0483208, 26.7668076, -82.8151245, 82.8151245
30: -40.6908493, 34.5612411, -40.6908493, 34.5612411, -75.2520905, 75.2520905
31: -43.5969238, 27.4388466, -43.5969238, 27.4388466, -71.0357666, 71.0357666
32: -45.7560997, 25.4878426, -45.7560997, 25.4878426, -71.2439423, 71.2439423
33: -64.4620056, 32.4007263, -64.4620056, 32.4007263, -96.8627243, 96.8627243
34: -53.7156601, 23.8152504, -53.7156601, 23.8152504, -77.5309143, 77.5309143
35: -55.1549835, 25.6195412, -55.1549835, 25.6195412, -80.7745209, 80.7745209
36: -52.9943619, 27.5708084, -52.9943619, 27.5708084, -80.5651703, 80.5651703
37: -73.6920776, 26.4342976, -73.6920776, 26.4342976, -100.1263733, 100.1263733
38: -64.1554718, 32.1214142, -64.1554718, 32.1214142, -96.2768784, 96.2768784
39: -76.0981598, 31.5777340, -76.0981598, 31.5777340, -107.6758881, 107.6758957
40: -60.8722305, 27.9689827, -60.8722305, 27.9689827, -88.8412170, 88.8412170
41: -44.6994591, 25.5963039, -44.6994591, 25.5963039, -70.2957611, 70.2957611
42: -32.4920807, 23.6799088, -32.4920807, 23.6799088, -56.1719894, 56.1719894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=117, inp2_unstable=117, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=571, inp2_unstable=571, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1569

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 695

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -40.1600134, upper bound: 40.2310716
time: 96.40 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -40.1891418, upper bound: 40.2328541
time: 86.17 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 184.58 seconds
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 184.58
Output dim: 4, lower bound: -40.1600134, upper bound: 40.2310716
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 184.58
Output dim: 4, lower bound: -40.1891418, upper bound: 40.2328541

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 125.59 + 585.79 = 711.38 seconds
