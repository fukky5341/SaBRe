## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 7)
Time budget: 3600 seconds
Split limit: 100
Threshold: 39.2744870991


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873)
1: (-44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5147743, 48.5147743)
2: (-35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6746597, 43.6746635)
3: (-46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2342491, 50.2342491)
4: (-36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7427902, 54.7427902)
5: (-49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0979233, 55.0979233)
6: (-43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734)
7: (-67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9659805, 55.9659729)
8: (-43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607)
9: (-21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806)
10: (-53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7385712, 68.7385712)
11: (-69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9733772, 46.9733734)
12: (-32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9903183, 59.9903183)
13: (-35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704)
14: (-105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.5284882, 89.5284729)
15: (-35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9814987, 56.9814987)
16: (-61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0701180, 55.0701103)
17: (-123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6229477, 82.6229477)
18: (-47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825)
19: (-40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6668015, 39.6667938)
20: (-31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844)
21: (-53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9202728, 49.9202728)
22: (-54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8721924, 55.8721924)
23: (-32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7988663, 38.7988663)
24: (-26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1127548, 44.1127586)
25: (-23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7622910, 41.7622833)
26: (-45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034)
27: (-45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015)
28: (-36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9219666, 48.9219627)
29: (-65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3150940, 50.3150940)
30: (-43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718)
31: (-41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588)
32: (-38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566)
33: (-19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7451019, 73.7450943)
34: (-28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6961746, 70.6961670)
35: (-18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6688843, 71.6688919)
36: (-27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8722839, 73.8722839)
37: (-14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0141983, 55.0141869)
38: (-33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6856689, 87.6856613)
39: (-19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5259857, 77.5259933)
40: (-22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9437256, 61.9437218)
41: (-26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076)
42: (-35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.60 + 98.61 = 101.21 seconds
status: Status.UNKNOWN
relational distance
Output dim: 37, lower bound: -39.3138009, upper bound: 39.3138009

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3130580, upper bound: 39.2849726
time: 68.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2849726, upper bound: 39.3130580
time: 81.00 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 149.56 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 149.56
Output dim: 37, lower bound: -39.3130580, upper bound: 39.2849726
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 149.56
Output dim: 37, lower bound: -39.2849726, upper bound: 39.3130580

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5202560, 48.5202332
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6811867, 43.6809731
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2531853, 50.2521935
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7725296, 54.7694016
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1214638, 55.1201553
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9889412, 55.9882126
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7422791, 68.7422485
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9733353, 46.9734344
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9913864, 59.9914474
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4898987, 89.4936981
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9852448, 56.9848671
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0863037, 55.0860863
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6515884, 82.6509247
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6673050, 39.6673279
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9243469, 49.9241524
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8722382, 55.8718987
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7985840, 38.8007736
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1225281, 44.1234398
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7634697, 41.7636337
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9376755, 48.9403496
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3166428, 50.3166695
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7424545, 73.7423630
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7103882, 70.7107315
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6799545, 71.6801453
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8733673, 73.8733597
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0205574, 55.0205498
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.7016907, 87.7020721
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5103378, 77.5087280
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9439507, 61.9439430
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1761

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3125322, upper bound: 39.2715041
time: 53.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2882609, upper bound: 39.2834601
time: 57.92 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5202332, 48.5202522
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6809731, 43.6811829
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2521935, 50.2531891
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7694016, 54.7725296
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1201515, 55.1214638
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9882164, 55.9889374
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7422485, 68.7422791
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9734421, 46.9733315
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9914474, 59.9913864
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4936829, 89.4898987
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9848709, 56.9852448
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0860901, 55.0863037
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6509323, 82.6515961
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6673279, 39.6673050
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9241486, 49.9243469
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8719025, 55.8722382
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.8007660, 38.7985840
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1234436, 44.1225204
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7636375, 41.7634697
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9403458, 48.9376717
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3166733, 50.3166504
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7423630, 73.7424545
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7107239, 70.7103958
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6801376, 71.6799469
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8733521, 73.8733673
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0205574, 55.0205574
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.7020721, 87.7016983
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5087357, 77.5103149
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9439430, 61.9439468
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1761

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2834601, upper bound: 39.2882609
time: 68.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2715041, upper bound: 39.3125322
time: 55.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 126.30 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 126.30
Output dim: 37, lower bound: -39.3125322, upper bound: 39.2715041
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 126.30
Output dim: 37, lower bound: -39.2882609, upper bound: 39.2834601
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 126.30
Output dim: 37, lower bound: -39.2834601, upper bound: 39.2882609
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 126.30
Output dim: 37, lower bound: -39.2715041, upper bound: 39.3125322

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5237503, 48.5237694
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6858139, 43.6855469
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2633743, 50.2620735
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7848282, 54.7807198
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1326790, 55.1310844
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -56.0024643, 56.0015335
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7450256, 68.7449417
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9806747, 46.9809113
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9919662, 59.9920502
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4784393, 89.4835663
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9847565, 56.9843750
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0962906, 55.0959549
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6672058, 82.6663589
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6683960, 39.6684303
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9299774, 49.9297180
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8733978, 55.8729477
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7943153, 38.7973671
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1249542, 44.1261787
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7629242, 41.7634354
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9425278, 48.9464302
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3182602, 50.3182945
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7447433, 73.7445374
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7181854, 70.7186203
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6865997, 71.6868744
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8749237, 73.8749084
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0264969, 55.0265083
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.7100525, 87.7105408
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5068207, 77.5048599
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9441833, 61.9441757
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 574

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2896188, upper bound: 39.2709124
time: 59.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3119360, upper bound: 39.2485050
time: 53.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5237808, 48.5237389
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6857605, 43.6855698
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2630692, 50.2622833
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7838516, 54.7813072
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1323891, 55.1313286
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -56.0022583, 56.0016861
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7449799, 68.7449951
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9808121, 46.9807358
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9919815, 59.9920349
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4797668, 89.4822388
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9847488, 56.9843750
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0961685, 55.0959816
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6670227, 82.6665421
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6684113, 39.6684151
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9299164, 49.9297333
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8732910, 55.8730545
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7951698, 38.7965012
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1250916, 44.1258774
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7632294, 41.7630844
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9435654, 48.9452095
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3182755, 50.3182793
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7446365, 73.7446442
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7182770, 70.7185287
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6865997, 71.6867905
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8749084, 73.8749237
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0265121, 55.0264893
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.7101440, 87.7104416
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5064545, 77.5052261
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9441757, 61.9441795
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 574

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2652834, upper bound: 39.2828662
time: 62.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2652834, upper bound: 39.2604779
time: 62.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5237350, 48.5237885
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6855698, 43.6857567
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2622910, 50.2630730
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7813034, 54.7838478
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1313210, 55.1323929
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -56.0016861, 56.0022583
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7449951, 68.7449799
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9807434, 46.9808083
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9920273, 59.9919891
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4822388, 89.4797668
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9843750, 56.9847488
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0959854, 55.0961723
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6665344, 82.6670227
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6684113, 39.6684113
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9297333, 49.9299126
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8730545, 55.8732910
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7965012, 38.7951698
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1258774, 44.1250877
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7630920, 41.7632370
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9452133, 48.9435654
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3182755, 50.3182755
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7446518, 73.7446289
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7185211, 70.7182846
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6867828, 71.6866150
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8749237, 73.8749084
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0264893, 55.0265160
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.7104492, 87.7101669
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5052338, 77.5064545
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9441757, 61.9441795
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 574

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2604779, upper bound: 39.2876664
time: 61.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2828662, upper bound: 39.2652834
time: 78.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5237656, 48.5237579
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6855469, 43.6858101
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2620773, 50.2633743
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7807236, 54.7848244
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1310921, 55.1326752
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -56.0015335, 56.0024643
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7449493, 68.7450256
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9809113, 46.9806709
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9920425, 59.9919739
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4835663, 89.4784393
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9843750, 56.9847527
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0959549, 55.0962868
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6663513, 82.6672058
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6684265, 39.6683960
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9297180, 49.9299812
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8729477, 55.8733978
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7973671, 38.7943153
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1261826, 44.1249580
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7634277, 41.7629242
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9464340, 48.9425316
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3182907, 50.3182602
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7445374, 73.7447357
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7186127, 70.7181931
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6868744, 71.6865921
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8749084, 73.8749237
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0265121, 55.0264969
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.7105408, 87.7100601
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5048676, 77.5068130
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9441757, 61.9441833
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 574

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2485050, upper bound: 39.3119360
time: 64.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2709124, upper bound: 39.2896188
time: 74.06 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 140.49 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 140.49
Output dim: 37, lower bound: -39.2896188, upper bound: 39.2709124
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 140.49
Output dim: 37, lower bound: -39.3119360, upper bound: 39.2485050
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 140.49
Output dim: 37, lower bound: -39.2652834, upper bound: 39.2828662
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 140.49
Output dim: 37, lower bound: -39.2652834, upper bound: 39.2604779
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 140.49
Output dim: 37, lower bound: -39.2604779, upper bound: 39.2876664
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 140.49
Output dim: 37, lower bound: -39.2828662, upper bound: 39.2652834
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 140.49
Output dim: 37, lower bound: -39.2485050, upper bound: 39.3119360
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 140.49
Output dim: 37, lower bound: -39.2709124, upper bound: 39.2896188

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5192642, 48.5177155
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6826706, 43.6811829
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2619247, 50.2599831
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7847672, 54.7804642
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1309662, 55.1287651
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9980011, 55.9953690
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7410660, 68.7396011
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9714279, 46.9684525
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9905472, 59.9907990
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4693146, 89.4721375
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9836884, 56.9830132
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0899887, 55.0874062
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6507416, 82.6441574
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6638145, 39.6622581
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9261093, 49.9244576
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8688850, 55.8668594
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7902527, 38.7916451
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1238403, 44.1252174
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7628021, 41.7632904
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9398880, 48.9428368
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3091888, 50.3060608
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7380295, 73.7395477
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7109985, 70.7132874
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6813812, 71.6830063
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8697815, 73.8710938
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0194473, 55.0212402
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.7021179, 87.7046509
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.4956818, 77.4965973
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9348221, 61.9374504
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1601

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2849002, upper bound: 39.2661817
time: 54.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2849002, upper bound: 39.2707505
time: 68.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5177002, 48.5192795
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6814499, 43.6824036
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2612839, 50.2606163
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7845688, 54.7806625
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1303558, 55.1293755
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9963074, 55.9970665
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7396851, 68.7409821
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9682083, 46.9716721
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9907303, 59.9906273
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4670258, 89.4744568
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9833908, 56.9833107
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0877457, 55.0896568
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6450043, 82.6498947
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6622276, 39.6638489
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9247208, 49.9258385
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8673134, 55.8684349
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7885895, 38.7933044
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1239929, 44.1250648
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7627716, 41.7633057
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9389420, 48.9437828
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3060303, 50.3092194
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7397614, 73.7378235
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7128601, 70.7114258
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6827240, 71.6816635
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8711090, 73.8697586
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0212326, 55.0194626
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.7041626, 87.7025986
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.4985657, 77.4937286
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9374466, 61.9348221
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1601

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3117735, upper bound: 39.2437672
time: 67.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3072185, upper bound: 39.2483432
time: 59.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5192947, 48.5176849
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6826096, 43.6812096
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2616196, 50.2601891
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7837906, 54.7810478
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1306763, 55.1290016
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9977875, 55.9955292
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7410126, 68.7396545
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9715652, 46.9682770
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9905624, 59.9907837
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4706573, 89.4708099
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9836884, 56.9830132
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0898819, 55.0874329
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6505585, 82.6443405
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6638298, 39.6622429
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9260330, 49.9244728
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8687782, 55.8669662
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7911072, 38.7907753
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1239624, 44.1249123
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7631073, 41.7629395
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9409103, 48.9416199
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3092041, 50.3060455
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7379227, 73.7396622
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7110901, 70.7131958
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6814117, 71.6829224
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8697662, 73.8711090
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0194626, 55.0212212
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.7022095, 87.7045517
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.4953308, 77.4969635
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9348221, 61.9374542
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1601

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2651216, upper bound: 39.2781491
time: 76.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2605496, upper bound: 39.2827039
time: 54.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5192490, 48.5177307
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6824265, 43.6813965
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2608261, 50.2609749
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7812500, 54.7835846
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1296158, 55.1300697
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9972229, 55.9960976
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7410278, 68.7396317
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9714890, 46.9683495
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9906082, 59.9907379
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4731293, 89.4683456
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9833145, 56.9833870
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0896835, 55.0876236
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6500702, 82.6448212
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6638298, 39.6622391
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9258652, 49.9246521
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8685417, 55.8671989
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7924385, 38.7894478
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1247559, 44.1241264
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7629547, 41.7630920
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9425583, 48.9399757
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3092041, 50.3060417
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7379379, 73.7396469
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7113342, 70.7129517
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6815796, 71.6827469
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8697662, 73.8710938
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0194473, 55.0212479
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.7024994, 87.7042694
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.4940948, 77.4981995
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9348221, 61.9374504
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1601

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2603161, upper bound: 39.2829524
time: 63.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2557426, upper bound: 39.2875040
time: 68.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5176849, 48.5192947
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6812057, 43.6826096
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2601852, 50.2616158
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7810516, 54.7837868
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1290054, 55.1306801
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9955292, 55.9977951
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7396469, 68.7410126
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9682693, 46.9715652
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9907913, 59.9905663
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4708099, 89.4706573
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9830093, 56.9836884
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0874405, 55.0898743
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6443329, 82.6505508
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6622429, 39.6638298
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9244766, 49.9260368
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8669624, 55.8687744
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7907791, 38.7911110
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1249084, 44.1239738
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7629395, 41.7631073
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9416122, 48.9409180
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3060455, 50.3092003
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7396622, 73.7379150
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7131958, 70.7110901
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6829224, 71.6814041
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8711090, 73.8697662
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0212173, 55.0194702
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.7045441, 87.7022247
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.4969788, 77.4953156
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9374466, 61.9348259
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1601

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2827039, upper bound: 39.2605496
time: 52.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2781491, upper bound: 39.2651216
time: 68.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5192795, 48.5177002
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6824112, 43.6814499
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2606125, 50.2612801
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7806625, 54.7845688
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1293716, 55.1303520
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9970703, 55.9963074
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7409744, 68.7396851
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9716721, 46.9682121
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9906235, 59.9907227
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4744415, 89.4670105
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9833145, 56.9833908
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0896530, 55.0877380
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6498871, 82.6450043
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6638451, 39.6622238
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9258347, 49.9247208
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8684349, 55.8673096
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7933044, 38.7885895
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1250610, 44.1239891
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7633057, 41.7627792
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9437790, 48.9389381
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3092194, 50.3060265
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7378235, 73.7397537
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7114258, 70.7128601
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6816711, 71.6827240
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8697510, 73.8711090
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0194626, 55.0212288
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.7025909, 87.7041779
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.4937439, 77.4985504
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9348221, 61.9374542
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1601

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2483432, upper bound: 39.3072185
time: 54.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2437672, upper bound: 39.3117735
time: 68.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5177155, 48.5192642
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6811905, 43.6826630
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2599869, 50.2619171
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7804642, 54.7847672
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1287613, 55.1309624
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9953766, 55.9980011
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7396011, 68.7410583
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9684525, 46.9714279
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9908066, 59.9905472
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4721527, 89.4693298
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9830093, 56.9836884
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0874100, 55.0899887
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6441650, 82.6507339
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6622581, 39.6638145
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9244614, 49.9261055
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8668556, 55.8688812
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7916412, 38.7902489
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1252136, 44.1238403
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7632904, 41.7627945
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9428329, 48.9398842
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3060608, 50.3091850
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7395554, 73.7380219
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7132874, 70.7109985
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6830139, 71.6813812
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8710938, 73.8697815
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0212326, 55.0194511
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.7046509, 87.7021179
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.4966125, 77.4956818
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9374466, 61.9348297
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1601

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2707505, upper bound: 39.2849002
time: 58.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2661817, upper bound: 39.2894567
time: 57.16 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 117.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 117.76
Output dim: 37, lower bound: -39.2849002, upper bound: 39.2661817
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 117.76
Output dim: 37, lower bound: -39.2849002, upper bound: 39.2707505
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 117.76
Output dim: 37, lower bound: -39.3117735, upper bound: 39.2437672
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 117.76
Output dim: 37, lower bound: -39.3072185, upper bound: 39.2483432
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 117.76
Output dim: 37, lower bound: -39.2651216, upper bound: 39.2781491
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 117.76
Output dim: 37, lower bound: -39.2605496, upper bound: 39.2827039
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 117.76
Output dim: 37, lower bound: -39.2603161, upper bound: 39.2829524
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 117.76
Output dim: 37, lower bound: -39.2557426, upper bound: 39.2875040
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 117.76
Output dim: 37, lower bound: -39.2827039, upper bound: 39.2605496
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 117.76
Output dim: 37, lower bound: -39.2781491, upper bound: 39.2651216
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 117.76
Output dim: 37, lower bound: -39.2483432, upper bound: 39.3072185
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 117.76
Output dim: 37, lower bound: -39.2437672, upper bound: 39.3117735
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 117.76
Output dim: 37, lower bound: -39.2707505, upper bound: 39.2849002
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 117.76
Output dim: 37, lower bound: -39.2661817, upper bound: 39.2894567

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5223351, 48.5208855
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6806412, 43.6792564
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2672768, 50.2658157
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7704620, 54.7663422
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1201401, 55.1191177
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9963303, 55.9937439
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7463531, 68.7449493
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9938202, 46.9911385
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9817200, 59.9813423
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4852982, 89.4877625
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9718323, 56.9724731
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0898170, 55.0872383
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6451263, 82.6390305
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6613960, 39.6600647
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9246635, 49.9230957
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8544273, 55.8542976
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7958145, 38.7974777
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1290474, 44.1301613
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7540131, 41.7556000
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9419250, 48.9451332
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3092270, 50.3061142
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7493515, 73.7507629
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7073822, 70.7093887
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6841049, 71.6856842
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8722839, 73.8735046
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0215569, 55.0234070
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6965027, 87.6985931
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5079575, 77.5084457
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9306335, 61.9327278
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2344590, upper bound: 39.2656757
time: 71.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2885951, upper bound: 39.2519927
time: 58.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5224342, 48.5207787
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6807327, 43.6791649
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2677498, 50.2653503
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7706528, 54.7661552
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1213150, 55.1179390
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9963760, 55.9936981
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7463989, 68.7448883
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9941101, 46.9908409
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9810944, 59.9819756
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4849472, 89.4881134
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9731445, 56.9711609
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0898247, 55.0872307
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6455994, 82.6385574
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6616173, 39.6598473
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9247475, 49.9230118
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8563194, 55.8524055
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7960815, 38.7972107
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1287880, 44.1304207
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7551041, 41.7545013
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9421844, 48.9448814
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3092346, 50.3061066
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7492294, 73.7508774
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7071075, 70.7096710
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6840591, 71.6857300
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8721924, 73.8735809
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0216179, 55.0233498
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6960449, 87.6990280
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5075302, 77.5088806
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9300995, 61.9332542
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2755981, upper bound: 39.2702438
time: 71.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2840317, upper bound: 39.2565636
time: 50.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5207710, 48.5224495
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6794357, 43.6804733
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2666512, 50.2664528
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7702560, 54.7665443
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1195297, 55.1197281
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9946365, 55.9954376
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7449799, 68.7463226
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9906006, 46.9943581
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9818954, 59.9811707
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4829941, 89.4900665
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9715424, 56.9727707
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0875664, 55.0894890
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6394043, 82.6447601
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6598091, 39.6616554
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9232750, 49.9244804
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8528557, 55.8558731
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7941551, 38.7991333
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1291924, 44.1300125
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7539902, 41.7556152
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9409866, 48.9460793
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3060760, 50.3092728
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7510834, 73.7490234
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7092438, 70.7075348
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6854477, 71.6843414
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8735962, 73.8721771
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0233421, 55.0216293
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6985474, 87.6965408
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5108261, 77.5055695
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9332581, 61.9300995
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2976641, upper bound: 39.2428998
time: 61.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3112735, upper bound: 39.2344590
time: 72.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5208702, 48.5223427
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6795273, 43.6803780
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2671089, 50.2659836
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7704468, 54.7663536
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1207047, 55.1185493
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9946747, 55.9953957
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7450256, 68.7462616
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9908905, 46.9940605
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9812622, 59.9818001
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4826279, 89.4904327
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9728546, 56.9714584
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0875740, 55.0894814
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6398621, 82.6442795
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6600304, 39.6614342
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9233665, 49.9243927
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8547478, 55.8539810
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7944183, 38.7988739
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1289406, 44.1302681
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7550888, 41.7545242
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9412384, 48.9458275
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3060760, 50.3092651
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7509689, 73.7491455
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7089539, 70.7078094
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6854019, 71.6843872
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8735352, 73.8722534
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0233955, 55.0215721
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6981201, 87.6969757
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5103989, 77.5060043
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9327240, 61.9306297
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2931092, upper bound: 39.2474749
time: 62.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3067189, upper bound: 39.2390359
time: 77.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5223656, 48.5208549
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6805954, 43.6792793
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2669868, 50.2660217
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7694778, 54.7669296
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1198502, 55.1193542
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9961243, 55.9938965
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7462921, 68.7449951
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9939575, 46.9909630
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9817352, 59.9813271
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4866257, 89.4864197
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9718323, 56.9724731
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0897026, 55.0872650
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6449432, 82.6392136
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6614113, 39.6600494
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9245872, 49.9231148
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8543205, 55.8544044
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7966766, 38.7966080
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1291771, 44.1298599
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7543182, 41.7552528
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9429550, 48.9439163
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3092422, 50.3060989
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7492447, 73.7508698
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7074738, 70.7093048
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6841354, 71.6856079
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8722534, 73.8735199
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0215797, 55.0233917
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6965942, 87.6984940
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5075912, 77.5088043
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9306335, 61.9327316
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2558181, upper bound: 39.2776436
time: 60.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2642518, upper bound: 39.2639638
time: 49.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5224648, 48.5207481
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6806870, 43.6791878
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2674446, 50.2655563
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7696686, 54.7667427
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1210403, 55.1181793
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9961624, 55.9938583
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7463531, 68.7449417
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9942474, 46.9906654
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9811096, 59.9819565
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4862747, 89.4867859
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9731445, 56.9711609
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0897102, 55.0872574
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6454163, 82.6387329
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6616325, 39.6598282
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9246788, 49.9230270
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8562126, 55.8525124
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7969398, 38.7963486
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1289177, 44.1301155
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7554169, 41.7541580
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9432144, 48.9436607
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3092499, 50.3060913
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7491226, 73.7509918
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7071991, 70.7095795
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6840744, 71.6856537
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8721924, 73.8735962
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0216331, 55.0233307
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6961670, 87.6989365
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5071640, 77.5092392
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9300995, 61.9332581
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2512425, upper bound: 39.2822018
time: 57.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2596813, upper bound: 39.2685341
time: 69.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5223122, 48.5209007
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6804123, 43.6794662
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2661934, 50.2668152
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7669449, 54.7694664
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1187973, 55.1204224
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9955521, 55.9944687
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7463226, 68.7449799
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9938812, 46.9910355
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9817810, 59.9812813
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4890976, 89.4839630
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9714508, 56.9728470
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0895119, 55.0874557
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6444702, 82.6396866
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6614189, 39.6600456
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9244118, 49.9232903
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8540916, 55.8546371
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7980042, 38.7952805
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1299629, 44.1290703
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7541733, 41.7553978
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9446030, 48.9422722
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3092499, 50.3060951
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7492599, 73.7508545
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7077179, 70.7090530
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6843033, 71.6854248
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8722687, 73.8735199
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0215569, 55.0234146
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6968689, 87.6982193
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5063705, 77.5100403
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9306335, 61.9327278
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2510130, upper bound: 39.2824484
time: 131.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2594461, upper bound: 39.2687693
time: 73.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5224190, 48.5208015
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6805038, 43.6793709
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2666664, 50.2663422
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7671280, 54.7692795
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1199722, 55.1192474
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9955978, 55.9944267
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7463684, 68.7449188
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9941864, 46.9907379
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9811554, 59.9819145
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4887314, 89.4843140
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9727631, 56.9715347
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0895195, 55.0874481
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6449432, 82.6392136
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6616402, 39.6598244
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9244957, 49.9232063
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8559837, 55.8527489
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7982712, 38.7950134
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1297112, 44.1293297
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7552643, 41.7543030
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9448624, 48.9420128
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3092575, 50.3060875
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7491379, 73.7509766
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7074432, 70.7093353
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6842575, 71.6854706
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8721924, 73.8735962
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0216179, 55.0233574
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6964417, 87.6986465
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5059433, 77.5104675
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9300995, 61.9332581
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2344590, upper bound: 39.2870030
time: 69.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2464372, upper bound: 39.2733395
time: 62.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5207481, 48.5224648
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6791916, 43.6806831
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2655525, 50.2674484
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7667389, 54.7696686
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1181870, 55.1210327
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9938583, 55.9961624
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7449341, 68.7463531
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9906616, 46.9942551
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9819565, 59.9811096
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4867783, 89.4862671
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9711609, 56.9731483
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0872612, 55.0897102
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6387329, 82.6454239
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6598244, 39.6616364
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9230309, 49.9246750
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8525124, 55.8562126
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7963448, 38.7969437
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1301155, 44.1289177
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7541580, 41.7554169
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9436569, 48.9432144
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3060913, 50.3092537
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7509918, 73.7491226
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7095795, 70.7071991
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6856613, 71.6840820
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8735962, 73.8721848
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0233269, 55.0216370
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6989136, 87.6961594
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5092392, 77.5071640
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9332581, 61.9301033
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2685341, upper bound: 39.2596813
time: 98.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2822018, upper bound: 39.2512425
time: 57.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5208549, 48.5223656
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6792831, 43.6805916
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2660255, 50.2669830
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7669296, 54.7694817
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1193619, 55.1198578
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9938965, 55.9961243
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7449951, 68.7462997
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9909668, 46.9939575
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9813232, 59.9817390
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4864273, 89.4866333
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9724731, 56.9718361
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0872688, 55.0896988
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6392059, 82.6449432
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6600456, 39.6614151
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9231148, 49.9245872
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8544044, 55.8543205
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7966080, 38.7966766
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1298561, 44.1291771
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7552490, 41.7543221
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9439163, 48.9429588
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3060989, 50.3092422
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7508774, 73.7492371
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7093048, 70.7074738
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6856003, 71.6841278
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8735352, 73.8722610
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0233879, 55.0215797
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6984863, 87.6966019
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5088120, 77.5075989
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9327240, 61.9306297
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2464372, upper bound: 39.2642517
time: 75.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2776436, upper bound: 39.2558181
time: 66.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5223427, 48.5208702
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6803818, 43.6795197
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2659798, 50.2671165
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7663574, 54.7704468
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1185532, 55.1207047
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9953995, 55.9946747
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7462616, 68.7450256
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9940643, 46.9908981
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9817963, 59.9812660
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4904251, 89.4826355
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9714508, 56.9728508
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0894890, 55.0875702
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6442871, 82.6398697
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6614342, 39.6600304
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9243889, 49.9233627
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8539848, 55.8547440
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7988739, 38.7944183
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1302681, 44.1289406
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7545166, 41.7550888
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9458237, 48.9412346
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3092651, 50.3060799
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7491531, 73.7509613
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7078094, 70.7089615
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6843796, 71.6854095
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8722534, 73.8735352
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0215721, 55.0233955
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6969604, 87.6981125
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5060043, 77.5103989
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9306335, 61.9327316
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2390359, upper bound: 39.3067189
time: 65.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2474749, upper bound: 39.2931092
time: 51.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5224495, 48.5207710
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6804733, 43.6794243
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2664528, 50.2666473
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7665482, 54.7702599
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1197281, 55.1195297
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9954376, 55.9946365
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7463226, 68.7449799
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9943542, 46.9906006
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9811707, 59.9818954
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4900742, 89.4829865
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9727631, 56.9715385
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0894890, 55.0875626
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6447601, 82.6393967
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6616554, 39.6598091
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9244804, 49.9232750
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8558769, 55.8528557
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7991333, 38.7941589
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1300087, 44.1291924
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7556152, 41.7539940
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9460831, 48.9409828
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3092728, 50.3060722
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7490311, 73.7510834
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7075348, 70.7092438
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6843491, 71.6854553
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8721771, 73.8736115
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0216331, 55.0233383
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6965332, 87.6985550
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5055771, 77.5108261
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9300995, 61.9332619
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2344590, upper bound: 39.3112735
time: 62.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2428998, upper bound: 39.2976641
time: 64.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5207787, 48.5224342
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6791611, 43.6807365
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2653542, 50.2677536
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7661514, 54.7706490
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1179428, 55.1213150
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9937057, 55.9963684
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7448883, 68.7464066
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9908447, 46.9941177
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9819794, 59.9810905
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4881210, 89.4849396
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9711609, 56.9731483
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0872307, 55.0898247
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6385498, 82.6455994
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6598473, 39.6616173
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9230156, 49.9247475
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8524055, 55.8563194
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7972107, 38.7960815
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1304207, 44.1287880
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7545013, 41.7551041
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9448776, 48.9421806
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3061066, 50.3092384
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7508774, 73.7492294
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7096710, 70.7071075
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6857224, 71.6840668
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8735962, 73.8722000
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0233498, 55.0216179
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6990356, 87.6960678
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5088730, 77.5075226
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9332581, 61.9301071
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2565636, upper bound: 39.2840316
time: 60.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2702438, upper bound: 39.2755981
time: 62.10 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 124.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2344590, upper bound: 39.2656757
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2885951, upper bound: 39.2519927
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2755981, upper bound: 39.2702438
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2840317, upper bound: 39.2565636
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2976641, upper bound: 39.2428998
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.3112735, upper bound: 39.2344590
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2931092, upper bound: 39.2474749
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.3067189, upper bound: 39.2390359
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2558181, upper bound: 39.2776436
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2642518, upper bound: 39.2639638
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2512425, upper bound: 39.2822018
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2596813, upper bound: 39.2685341
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2510130, upper bound: 39.2824484
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2594461, upper bound: 39.2687693
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2344590, upper bound: 39.2870030
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2464372, upper bound: 39.2733395
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2685341, upper bound: 39.2596813
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2822018, upper bound: 39.2512425
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2464372, upper bound: 39.2642517
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2776436, upper bound: 39.2558181
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2390359, upper bound: 39.3067189
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2474749, upper bound: 39.2931092
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2344590, upper bound: 39.3112735
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2428998, upper bound: 39.2976641
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2565636, upper bound: 39.2840316
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 124.64
Output dim: 37, lower bound: -39.2702438, upper bound: 39.2755981
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 124.64
Output dim: 37, lower bound: -39.2661817, upper bound: 39.2894567

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 101.21 + 3593.10 = 3694.31 seconds
