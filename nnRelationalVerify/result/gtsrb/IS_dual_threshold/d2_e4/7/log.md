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
execution time: IAR + RelationalAnalysis = 2.37 + 97.93 = 100.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 37, lower bound: -39.3138009, upper bound: 39.3138009

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 1110
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 1140
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1211
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1223
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1164
type: B, layer: 1, pos: 1164
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1142
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1212
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1095
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1156
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1208
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 1227
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1109
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1218
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1127
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1104
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1120
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 729

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3107079, upper bound: 39.2836883
time: 64.84 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3107079, upper bound: 39.3107078
time: 63.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 128.85 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 128.85
Output dim: 37, lower bound: -39.3107079, upper bound: 39.2836883
IS_A2, status: Status.UNKNOWN, split count: 1, time: 128.85
Output dim: 37, lower bound: -39.3107079, upper bound: 39.3107078

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -63.4468346, 8.5304680, -63.5201836, 8.6015177, -72.0483551, 72.0506516
1: -44.6072044, 6.6037846, -44.6620293, 6.6748724, -48.3697510, 48.3469963
2: -35.3698120, 10.6655025, -35.4217796, 10.7178125, -43.5358658, 43.5361176
3: -46.0125504, 7.4273577, -46.0672913, 7.4825344, -50.0881882, 50.0879440
4: -36.2491913, 19.6279068, -36.2867355, 19.6640854, -54.6484756, 54.6367760
5: -49.2076340, 11.0938301, -49.2607079, 11.1357718, -54.9642601, 54.9758568
6: -43.4943008, 19.2155361, -43.5223007, 19.2117062, -62.7060089, 62.7378387
7: -66.8612976, -0.3969288, -66.9338303, -0.3263111, -55.7666397, 55.7685585
8: -42.8251724, 24.9471970, -42.9064522, 25.0369873, -67.8621597, 67.8536530
9: -21.6449947, 16.4727173, -21.6652069, 16.4933052, -38.1382980, 38.1379242
10: -53.3033867, 17.4561062, -53.3396416, 17.4889221, -68.6442642, 68.6429138
11: -69.6349945, -11.5733147, -69.7002563, -11.5662451, -46.8102341, 46.8765869
12: -32.5819473, 30.0566673, -32.6249580, 30.1010036, -59.8752594, 59.8758621
13: -35.8591309, 37.4158859, -35.8780060, 37.4443283, -73.3034592, 73.2938919
14: -105.5168610, -11.0504494, -105.6074524, -10.9800129, -89.2956085, 89.3146667
15: -35.3313637, 21.9023190, -35.3561668, 21.9213219, -56.9072266, 56.8974495
16: -61.0581131, 2.3058567, -61.0939102, 2.3277168, -54.9756355, 54.9865303
17: -123.1228104, -17.6582375, -123.1735611, -17.6169624, -82.5131073, 82.5342102
18: -47.0121040, 24.1414833, -47.0775108, 24.2021885, -71.2142944, 71.2189941
19: -40.2570267, 1.7368941, -40.2753220, 1.7535601, -39.6279678, 39.6406593
20: -31.6924686, 5.3852158, -31.7175274, 5.3988376, -37.0913048, 37.1027451
21: -53.2704697, 0.2035265, -53.3014717, 0.2164650, -49.8467445, 49.8667107
22: -54.0522118, 6.0882502, -54.0810204, 6.1269207, -55.7497749, 55.7445068
23: -32.8415680, 8.2794590, -32.8562851, 8.2893600, -38.7343254, 38.7674026
24: -26.0941753, 18.4710751, -26.1176815, 18.4999924, -44.0353889, 44.0328293
25: -23.4279270, 19.6879234, -23.4668560, 19.7465382, -41.6233673, 41.6050034
26: -44.9758377, 25.2373276, -45.0500336, 25.2756634, -70.2515030, 70.2873611
27: -45.7630501, 10.9623756, -45.7833481, 10.9753389, -56.7383881, 56.7457237
28: -36.2023697, 14.3875103, -36.2234344, 14.4066286, -48.8418274, 48.8693848
29: -65.3661652, -6.1483669, -65.4023285, -6.1278028, -50.2289047, 50.2467232
30: -43.8240891, 14.4285612, -43.8478622, 14.4566383, -58.2807274, 58.2764244
31: -41.6330261, 2.8692498, -41.6643486, 2.8930187, -44.5260468, 44.5335999
32: -38.7813797, 22.6104507, -38.8022919, 22.6168842, -61.3982620, 61.4127426
33: -19.6794014, 60.1222725, -19.7339287, 60.1918488, -73.5518341, 73.5433350
34: -28.2301674, 47.6161766, -28.2801304, 47.6676331, -70.5504990, 70.5488586
35: -18.5915680, 56.1451263, -18.6073208, 56.1739426, -71.6228943, 71.5950623
36: -27.2828484, 48.3996048, -27.3353653, 48.4260025, -73.7595825, 73.7874603
37: -14.7328243, 48.4548988, -14.8062916, 48.5047569, -54.8305855, 54.8595276
38: -33.2820168, 57.8432693, -33.3176422, 57.8613586, -87.6016083, 87.6222916
39: -19.6967812, 65.9295654, -19.7042599, 65.9662933, -77.4481049, 77.4315491
40: -22.8643703, 42.3748932, -22.9029427, 42.4004173, -61.8499947, 61.8605919
41: -26.0595818, 26.3692913, -26.0876198, 26.3618011, -52.4213829, 52.4569092
42: -35.6518517, 19.6287003, -35.6738892, 19.6299419, -55.2817917, 55.3025894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=258, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1110
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1140
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1164
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1163
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1156
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 1227
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1109
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1105
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1104
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1120
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1168
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1597

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2892786, upper bound: 39.2771867
time: 60.73 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3042104, upper bound: 39.2771867
time: 67.13 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -63.6098175, 8.6108551, -63.6122932, 8.6118526, -72.2216721, 72.2231445
1: -44.7285881, 6.6882753, -44.7309952, 6.6893091, -48.4759750, 48.5115471
2: -35.4862289, 10.7280264, -35.4884224, 10.7293291, -43.6159286, 43.6699753
3: -46.1367340, 7.4965000, -46.1393890, 7.4982567, -50.1900482, 50.2285576
4: -36.3293343, 19.6733360, -36.3312378, 19.6743813, -54.7356796, 54.7518845
5: -49.3236313, 11.1460085, -49.3258286, 11.1474905, -55.0686569, 55.0928650
6: -43.5465965, 19.2296352, -43.5556068, 19.2316189, -62.7782135, 62.7852402
7: -67.0221252, -0.3097744, -67.0278473, -0.3079891, -55.8743744, 55.9566803
8: -43.0063400, 25.0526028, -43.0085373, 25.0547867, -68.0611267, 68.0611420
9: -21.6855412, 16.5028954, -21.6865349, 16.5045719, -38.1901131, 38.1894302
10: -53.3766212, 17.5043163, -53.3783264, 17.5062141, -68.6968460, 68.7329712
11: -69.7550125, -11.5486012, -69.7699051, -11.5468597, -46.9016724, 46.9580345
12: -32.6367340, 30.1496696, -32.6386108, 30.1513557, -59.9879799, 59.9843559
13: -35.8894424, 37.4604416, -35.8904343, 37.4622650, -73.3517075, 73.3508759
14: -105.7122879, -10.9609509, -105.7151947, -10.9577751, -89.3898163, 89.5184021
15: -35.3834381, 21.9337826, -35.3854523, 21.9374580, -56.9678345, 56.9892578
16: -61.1266556, 2.3434210, -61.1343269, 2.3449230, -55.0333061, 55.0603294
17: -123.2302017, -17.6016159, -123.2335815, -17.5994110, -82.5928421, 82.6089401
18: -47.0981293, 24.2738571, -47.1010971, 24.2757092, -71.3738403, 71.3749542
19: -40.2905960, 1.7592149, -40.2924500, 1.7597318, -39.6684647, 39.6597519
20: -31.7318783, 5.4132576, -31.7327805, 5.4145346, -37.1464119, 37.1460381
21: -53.3252945, 0.2271309, -53.3297958, 0.2281694, -49.9028931, 49.9141388
22: -54.1039772, 6.1702051, -54.1061630, 6.1768465, -55.8620071, 55.8267784
23: -32.8656769, 8.2979183, -32.8672714, 8.2984285, -38.8315506, 38.7902756
24: -26.1298199, 18.5332355, -26.1309662, 18.5346909, -44.1114883, 44.1081734
25: -23.4780674, 19.8117962, -23.4791622, 19.8180008, -41.7543869, 41.7033691
26: -45.0758743, 25.3228836, -45.0781441, 25.3245964, -70.4004669, 70.4010315
27: -45.8006363, 10.9862537, -45.8024559, 10.9869995, -56.7876358, 56.7887115
28: -36.2316971, 14.4221954, -36.2323990, 14.4232187, -48.9499779, 48.9130974
29: -65.4382172, -6.1174097, -65.4403534, -6.1162720, -50.2870331, 50.3104668
30: -43.8576622, 14.4816723, -43.8585091, 14.4834795, -58.3411407, 58.3401794
31: -41.6742897, 2.9032092, -41.6750641, 2.9040456, -44.5783348, 44.5782738
32: -38.8131523, 22.6300964, -38.8198624, 22.6309395, -61.4440918, 61.4499588
33: -19.7508621, 60.2761307, -19.7530861, 60.2809486, -73.7362976, 73.6452942
34: -28.2997055, 47.7307320, -28.3018684, 47.7317924, -70.6900024, 70.6497650
35: -18.6207905, 56.2177963, -18.6225014, 56.2198563, -71.6615143, 71.6431885
36: -27.3526764, 48.4585648, -27.3548927, 48.4592285, -73.8670349, 73.8343887
37: -14.8347435, 48.5706673, -14.8393822, 48.5725136, -55.0024071, 54.9172935
38: -33.3419266, 57.8814125, -33.3453102, 57.8826637, -87.6748581, 87.6675720
39: -19.7192326, 66.0127563, -19.7214413, 66.0185852, -77.5155792, 77.4724884
40: -22.9237900, 42.4277115, -22.9267159, 42.4290962, -61.9368820, 61.9253731
41: -26.1164665, 26.3728981, -26.1258640, 26.3735485, -52.4900131, 52.4987640
42: -35.6928864, 19.6448479, -35.6994820, 19.6464367, -55.3393250, 55.3443298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=258, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1140
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1223
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1164
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1212
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1208
type: A, layer: 1, pos: 1208
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1104
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1597

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2892786, upper bound: 39.3042103
time: 60.30 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3042104, upper bound: 39.3042103
time: 64.06 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 126.38 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 126.38
Output dim: 37, lower bound: -39.2892786, upper bound: 39.2771867
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 126.38
Output dim: 37, lower bound: -39.3042104, upper bound: 39.2771867
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 126.38
Output dim: 37, lower bound: -39.2892786, upper bound: 39.3042103
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 126.38
Output dim: 37, lower bound: -39.3042104, upper bound: 39.3042103

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -63.4102364, 8.5185919, -63.4598160, 8.5817719, -71.9920044, 71.9784088
1: -44.5808868, 6.5957060, -44.6186256, 6.6614742, -48.3161774, 48.2825012
2: -35.3583946, 10.6533356, -35.4028397, 10.6975508, -43.4379768, 43.4393959
3: -46.0060883, 7.4118986, -46.0565796, 7.4567585, -49.9589653, 49.9658394
4: -36.2170258, 19.6121426, -36.2334099, 19.6378689, -54.5477371, 54.5224915
5: -49.2003059, 11.0754509, -49.2485275, 11.1051178, -54.7591095, 54.7787895
6: -43.4871750, 19.1537075, -43.5105133, 19.1076984, -62.5948715, 62.6642227
7: -66.8414383, -0.4105949, -66.9007339, -0.3490028, -55.6447868, 55.6428452
8: -42.7900543, 24.9307556, -42.8481140, 25.0096188, -67.7996750, 67.7788696
9: -21.6266651, 16.4645443, -21.6349373, 16.4797630, -38.1064301, 38.0994797
10: -53.2423019, 17.4391365, -53.2371292, 17.4607582, -68.5502243, 68.5186462
11: -69.6016083, -11.5795326, -69.6442642, -11.5765123, -46.6916161, 46.7376785
12: -32.5603180, 30.0467033, -32.5888023, 30.0846519, -59.7340775, 59.7269173
13: -35.8492584, 37.3496132, -35.8616371, 37.3336182, -73.1828766, 73.2112503
14: -105.4018860, -11.0567818, -105.4150085, -10.9905434, -89.1717224, 89.1141281
15: -35.2927399, 21.8907204, -35.2925415, 21.9020500, -56.7990570, 56.7717247
16: -61.0163994, 2.2886581, -61.0240746, 2.2991695, -54.9266472, 54.9193459
17: -123.0230255, -17.6747742, -123.0058060, -17.6441975, -82.5057526, 82.4664001
18: -46.9460716, 24.1337242, -46.9667358, 24.1892796, -71.1353531, 71.1004639
19: -40.2302475, 1.7281065, -40.2305107, 1.7390313, -39.6762848, 39.6753922
20: -31.6734085, 5.3819613, -31.6860161, 5.3934426, -37.0668526, 37.0679779
21: -53.2225227, 0.1980829, -53.2213211, 0.2074881, -49.8450012, 49.8352280
22: -53.9772568, 6.0822773, -53.9553604, 6.1171122, -55.8256454, 55.7720375
23: -32.8175583, 8.2743320, -32.8162994, 8.2808599, -38.7213135, 38.7421951
24: -26.0361843, 18.4678879, -26.0208473, 18.4947491, -43.9434090, 43.9043465
25: -23.4018021, 19.6842747, -23.4234657, 19.7405357, -41.6396179, 41.6056480
26: -44.9031830, 25.2309570, -44.9284210, 25.2650394, -70.1326599, 70.1218109
27: -45.7065582, 10.9575291, -45.6893311, 10.9673023, -56.6738586, 56.6468582
28: -36.1796875, 14.3823004, -36.1858711, 14.3979702, -48.8237839, 48.8391533
29: -65.2949524, -6.1543512, -65.2830200, -6.1376524, -50.2070999, 50.1780014
30: -43.7692070, 14.4223919, -43.7566528, 14.4465294, -58.2157364, 58.1790466
31: -41.6136551, 2.8619456, -41.6320114, 2.8810143, -44.4946709, 44.4939575
32: -38.7592163, 22.5575867, -38.7653618, 22.5303650, -61.2895813, 61.3229485
33: -19.6627159, 60.0613899, -19.7062912, 60.0895615, -73.4191895, 73.4415588
34: -28.2129822, 47.5823441, -28.2516594, 47.6109924, -70.4355774, 70.4457550
35: -18.5790825, 56.0918808, -18.5866280, 56.0845757, -71.5323715, 71.5330811
36: -27.2691345, 48.3267670, -27.3126850, 48.3038406, -73.6193008, 73.6875687
37: -14.7085896, 48.4292755, -14.7661304, 48.4617233, -54.8119240, 54.8436890
38: -33.2636223, 57.7578239, -33.2870712, 57.7180786, -87.4501495, 87.5187225
39: -19.6745415, 65.8531036, -19.6673851, 65.8380890, -77.3022614, 77.3246307
40: -22.8396244, 42.3283081, -22.8619461, 42.3222504, -61.7136459, 61.7390442
41: -26.0511208, 26.3227234, -26.0735912, 26.2856827, -52.3368034, 52.3963165
42: -35.6426392, 19.5864372, -35.6585236, 19.5591507, -55.2017899, 55.2449608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1164
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1163
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1222
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1212
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: A, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1156
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1208
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1109
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1105
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1598

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2873741, upper bound: 39.2484861
time: 68.94 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2873741, upper bound: 39.2753279
time: 84.46 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -63.4352570, 8.5280685, -63.5484314, 8.6508331, -72.0860901, 72.0764999
1: -44.6043015, 6.6019449, -44.6772385, 6.7677374, -48.4576111, 48.3543243
2: -35.3614578, 10.6633329, -35.4205894, 10.7417326, -43.5475731, 43.5390244
3: -46.0110092, 7.4248981, -46.0918236, 7.5065212, -50.0990982, 50.1253510
4: -36.2415199, 19.6248608, -36.3028793, 19.7499466, -54.7300034, 54.6453705
5: -49.2057076, 11.0912113, -49.2780685, 11.1624393, -54.9794693, 55.0072136
6: -43.4918175, 19.2082176, -43.6367073, 19.2241688, -62.7159882, 62.8449249
7: -66.8518677, -0.3990097, -66.9372940, -0.2728119, -55.8055763, 55.7690392
8: -42.8166885, 24.9442787, -42.9151764, 25.1043758, -67.9210663, 67.8594513
9: -21.6418018, 16.4708595, -21.7190838, 16.5656776, -38.2074814, 38.1899414
10: -53.2962837, 17.4526520, -53.3534393, 17.6320763, -68.7824249, 68.6474915
11: -69.6300507, -11.5746880, -69.7131958, -11.4629250, -46.9128571, 46.8807526
12: -32.5789299, 30.0531864, -32.6579590, 30.1206379, -59.8951454, 59.9117851
13: -35.8571472, 37.4059639, -36.0372086, 37.4533768, -73.3105240, 73.4431763
14: -105.5026245, -11.0521441, -105.6323929, -10.7870378, -89.4832153, 89.3172379
15: -35.3274307, 21.8999290, -35.3703423, 22.0364799, -57.0172195, 56.9011993
16: -61.0523682, 2.3031750, -61.1285667, 2.4379587, -55.0801468, 55.0099564
17: -123.1116562, -17.6612339, -123.1868210, -17.4193478, -82.7020416, 82.5154572
18: -47.0034027, 24.1396523, -47.0859528, 24.3466530, -71.3500519, 71.2256012
19: -40.2531204, 1.7351398, -40.2963943, 1.7999926, -39.6722717, 39.6560593
20: -31.6884823, 5.3843274, -31.7489414, 5.4425068, -37.1309891, 37.1332703
21: -53.2639465, 0.2020102, -53.3332367, 0.3055325, -49.9273682, 49.8951836
22: -54.0437775, 6.0866241, -54.1040649, 6.2191448, -55.8346901, 55.7548180
23: -32.8372421, 8.2781982, -32.8752861, 8.3517256, -38.7923813, 38.7830048
24: -26.0861664, 18.4703140, -26.1338806, 18.5843029, -44.1161270, 44.0469170
25: -23.4221268, 19.6868000, -23.5036068, 19.8058701, -41.6777649, 41.6384354
26: -44.9666519, 25.2356052, -45.0750771, 25.4143105, -70.3809662, 70.3106842
27: -45.7544022, 10.9610462, -45.8016968, 11.0690594, -56.8234634, 56.7627411
28: -36.1977615, 14.3863945, -36.2432098, 14.4480410, -48.8809776, 48.8860703
29: -65.3579102, -6.1498709, -65.4175110, -6.0251055, -50.3250160, 50.2452850
30: -43.8143692, 14.4270525, -43.8664436, 14.5597677, -58.3741379, 58.2934952
31: -41.6297760, 2.8675995, -41.6945267, 2.9538193, -44.5835953, 44.5621262
32: -38.7777214, 22.6062164, -38.9119415, 22.6299591, -61.4076805, 61.5181580
33: -19.6756973, 60.1151352, -19.9271889, 60.1932831, -73.5435715, 73.7303772
34: -28.2260075, 47.6117935, -28.3724442, 47.6816368, -70.5510254, 70.6365204
35: -18.5885315, 56.1384888, -18.7508621, 56.1773033, -71.6170044, 71.7322998
36: -27.2803688, 48.3906708, -27.4937134, 48.4227562, -73.7467422, 73.9376984
37: -14.7283173, 48.4517097, -14.9450493, 48.5055771, -54.8146515, 54.9997215
38: -33.2787857, 57.8320312, -33.5206718, 57.8735199, -87.5992584, 87.8152084
39: -19.6929893, 65.9196777, -19.9083881, 65.9588928, -77.4249191, 77.6282349
40: -22.8592758, 42.3714485, -23.0154419, 42.4072342, -61.8372459, 61.9589539
41: -26.0573654, 26.3618126, -26.2043018, 26.3743572, -52.4317245, 52.5661163
42: -35.6498985, 19.6243038, -35.7647209, 19.6446609, -55.2945595, 55.3890228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 1195
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1223
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1164
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1163
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1160
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1222
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1224
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1212
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 1095
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1156
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1208
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1208
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1109
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1105
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1218
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1104
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1120
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1121
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 769

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1598

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2873741, upper bound: 39.2484861
time: 60.75 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3023597, upper bound: 39.2753279
time: 52.27 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -63.5731697, 8.5989761, -63.5519104, 8.5921364, -72.1653061, 72.1508865
1: -44.7022629, 6.6801634, -44.6876144, 6.6759214, -48.4223747, 48.4470367
2: -35.4748039, 10.7158575, -35.4694824, 10.7090588, -43.5179977, 43.5732269
3: -46.1302795, 7.4810276, -46.1286812, 7.4724512, -50.0608215, 50.1064568
4: -36.2971954, 19.6576004, -36.2779388, 19.6481495, -54.6349602, 54.6375999
5: -49.3162880, 11.1276350, -49.3136520, 11.1167955, -54.8634949, 54.8958168
6: -43.5394554, 19.1677952, -43.5438004, 19.1276188, -62.6670761, 62.7115936
7: -67.0022430, -0.3234634, -66.9947510, -0.3306770, -55.7525253, 55.8309746
8: -42.9712601, 25.0361671, -42.9502220, 25.0273724, -67.9986343, 67.9863892
9: -21.6672039, 16.4947147, -21.6562595, 16.4910259, -38.1582298, 38.1509743
10: -53.3155670, 17.4873695, -53.2758598, 17.4780293, -68.6028366, 68.6087112
11: -69.7215958, -11.5548372, -69.7139053, -11.5571184, -46.7830124, 46.8190956
12: -32.6150665, 30.1396885, -32.6024246, 30.1350212, -59.8467979, 59.8353920
13: -35.8795700, 37.3941803, -35.8740654, 37.3515549, -73.2311249, 73.2682495
14: -105.5973587, -10.9672804, -105.5227737, -10.9682369, -89.2659454, 89.3178253
15: -35.3448181, 21.9222069, -35.3218117, 21.9181862, -56.8596954, 56.8635406
16: -61.0849533, 2.3262262, -61.0644798, 2.3163834, -54.9843559, 54.9931526
17: -123.1304321, -17.6180496, -123.0658112, -17.6266212, -82.5854721, 82.5411530
18: -47.0320969, 24.2660694, -46.9903412, 24.2628098, -71.2949066, 71.2564087
19: -40.2638321, 1.7504325, -40.2476273, 1.7452054, -39.7167587, 39.6944923
20: -31.7127972, 5.4100008, -31.7012520, 5.4091349, -37.1219330, 37.1112518
21: -53.2773323, 0.2216845, -53.2496529, 0.2192011, -49.9011726, 49.8826866
22: -54.0290527, 6.1642189, -53.9805412, 6.1670332, -55.9379311, 55.8542671
23: -32.8416672, 8.2927780, -32.8272896, 8.2899332, -38.8185425, 38.7650642
24: -26.0718536, 18.5300522, -26.0341454, 18.5294380, -44.0195389, 43.9797020
25: -23.4519444, 19.8081322, -23.4357662, 19.8119984, -41.7706337, 41.7039871
26: -45.0032425, 25.3164902, -44.9565277, 25.3139725, -70.2797470, 70.2355957
27: -45.7441788, 10.9814005, -45.7084274, 10.9789772, -56.7231560, 56.6898270
28: -36.2090225, 14.4169712, -36.1948051, 14.4145555, -48.9319077, 48.8828773
29: -65.3669815, -6.1234102, -65.3210297, -6.1261568, -50.2653236, 50.2418633
30: -43.8028069, 14.4754877, -43.7673187, 14.4733963, -58.2762032, 58.2428055
31: -41.6549225, 2.8959107, -41.6427193, 2.8920412, -44.5469627, 44.5386314
32: -38.7910042, 22.5772705, -38.7829094, 22.5444374, -61.3354416, 61.3601799
33: -19.7342129, 60.2152443, -19.7254181, 60.1786613, -73.6036987, 73.5435486
34: -28.2825356, 47.6968689, -28.2733994, 47.6751175, -70.5750580, 70.5466385
35: -18.6083107, 56.1645508, -18.6017990, 56.1304703, -71.5709610, 71.5812912
36: -27.3389435, 48.3857307, -27.3321800, 48.3370476, -73.7266998, 73.7345047
37: -14.8105278, 48.5450592, -14.7992020, 48.5294952, -54.9837341, 54.9014549
38: -33.3235016, 57.7959824, -33.3146667, 57.7393532, -87.5234528, 87.5640106
39: -19.6970291, 65.9362946, -19.6845951, 65.8904114, -77.3697205, 77.3655396
40: -22.8990822, 42.3811035, -22.8857307, 42.3509483, -61.8004608, 61.8038330
41: -26.1079998, 26.3263493, -26.1118164, 26.2974453, -52.4054451, 52.4381638
42: -35.6836700, 19.6025887, -35.6841393, 19.5756493, -55.2593193, 55.2867279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1140
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1223
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1164
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1163
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: A, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1208
type: A, layer: 1, pos: 1208
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1227
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 1105
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1104
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1598

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2873741, upper bound: 39.2755063
time: 72.58 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2873741, upper bound: 39.3023596
time: 63.86 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -63.5982132, 8.6084995, -63.6405334, 8.6612473, -72.2594604, 72.2490311
1: -44.7256699, 6.6864214, -44.7462082, 6.7821827, -48.5638351, 48.5188599
2: -35.4778709, 10.7258568, -35.4872513, 10.7532558, -43.6276093, 43.6728973
3: -46.1352005, 7.4940357, -46.1639175, 7.5222149, -50.2009964, 50.2659302
4: -36.3216400, 19.6702995, -36.3473930, 19.7602177, -54.8172112, 54.7605019
5: -49.3217125, 11.1433649, -49.3431931, 11.1741676, -55.0838585, 55.1242294
6: -43.5440826, 19.2223110, -43.6699753, 19.2441330, -62.7882156, 62.8922882
7: -67.0126801, -0.3118916, -67.0312805, -0.2544937, -55.9133301, 55.9571686
8: -42.9978905, 25.0496979, -43.0172577, 25.1221600, -68.1200485, 68.0669556
9: -21.6823387, 16.5010242, -21.7403908, 16.5769348, -38.2592735, 38.2414169
10: -53.3695374, 17.5008736, -53.3921280, 17.6493683, -68.8349915, 68.7375336
11: -69.7500839, -11.5499907, -69.7828522, -11.4435263, -47.0042877, 46.9622002
12: -32.6336784, 30.1461411, -32.6715851, 30.1710072, -60.0078316, 60.0202904
13: -35.8874283, 37.4505196, -36.0496140, 37.4713211, -73.3587494, 73.5001373
14: -105.6980438, -10.9626570, -105.7401657, -10.7648115, -89.5773621, 89.5209885
15: -35.3795166, 21.9314194, -35.3997040, 22.0526333, -57.0778351, 56.9928284
16: -61.1208992, 2.3407440, -61.1689644, 2.4551783, -55.1378326, 55.0837822
17: -123.2190323, -17.6045494, -123.2467651, -17.4018211, -82.7817383, 82.5901794
18: -47.0894012, 24.2719994, -47.1095314, 24.4201622, -71.5095673, 71.3815308
19: -40.2866898, 1.7574592, -40.3135452, 1.8061619, -39.7127228, 39.6751556
20: -31.7278748, 5.4123735, -31.7641907, 5.4581823, -37.1860580, 37.1765633
21: -53.3187447, 0.2256327, -53.3615837, 0.3172512, -49.9834976, 49.9426155
22: -54.0955124, 6.1685829, -54.1292496, 6.2690573, -55.9469566, 55.8371086
23: -32.8613396, 8.2966480, -32.8862762, 8.3607893, -38.8895836, 38.8058434
24: -26.1218147, 18.5324726, -26.1471691, 18.6189804, -44.1922302, 44.1222649
25: -23.4722557, 19.8106766, -23.5159092, 19.8773327, -41.8087692, 41.7368011
26: -45.0666580, 25.3211441, -45.1032333, 25.4632282, -70.5298843, 70.4243774
27: -45.7920074, 10.9849224, -45.8207855, 11.0807152, -56.8727226, 56.8057098
28: -36.2270813, 14.4210739, -36.2521820, 14.4646235, -48.9890862, 48.9297638
29: -65.4299469, -6.1189156, -65.4555206, -6.0136108, -50.3831520, 50.3090782
30: -43.8479538, 14.4801559, -43.8771095, 14.5865879, -58.4345398, 58.3572655
31: -41.6710587, 2.9015479, -41.7052536, 2.9648433, -44.6359024, 44.6068001
32: -38.8094864, 22.6258621, -38.9295044, 22.6440277, -61.4535141, 61.5553665
33: -19.7471733, 60.2689514, -19.9463310, 60.2824059, -73.7280655, 73.8323441
34: -28.2955399, 47.7263565, -28.3941498, 47.7457809, -70.6905365, 70.7373657
35: -18.6177788, 56.2111206, -18.7660370, 56.2231827, -71.6555862, 71.7805099
36: -27.3501472, 48.4496498, -27.5132103, 48.4559669, -73.8541641, 73.9846191
37: -14.8302269, 48.5674858, -14.9781494, 48.5733376, -54.9864883, 55.0574837
38: -33.3386765, 57.8701820, -33.5482750, 57.8947716, -87.6725540, 87.8604431
39: -19.7154388, 66.0028534, -19.9255867, 66.0112000, -77.4924088, 77.6691360
40: -22.9186878, 42.4242401, -23.0391769, 42.4359322, -61.9241333, 62.0237198
41: -26.1142349, 26.3654060, -26.2425289, 26.3861179, -52.5003510, 52.6079330
42: -35.6909142, 19.6404305, -35.7902908, 19.6611671, -55.3520813, 55.4307213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1140
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1223
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1164
type: A, layer: 1, pos: 1164
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1224
type: B, layer: 1, pos: 1222
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1212
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1156
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1208
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1227
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1105
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1104
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1598

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3023597, upper bound: 39.2755063
time: 54.79 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2873741, upper bound: 39.3023596
time: 52.13 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 109.00 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 109.00
Output dim: 37, lower bound: -39.2873741, upper bound: 39.2484861
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 109.00
Output dim: 37, lower bound: -39.2873741, upper bound: 39.2753279
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 109.00
Output dim: 37, lower bound: -39.2873741, upper bound: 39.2484861
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 109.00
Output dim: 37, lower bound: -39.3023597, upper bound: 39.2753279
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 109.00
Output dim: 37, lower bound: -39.2873741, upper bound: 39.2755063
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 109.00
Output dim: 37, lower bound: -39.2873741, upper bound: 39.3023596
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 109.00
Output dim: 37, lower bound: -39.3023597, upper bound: 39.2755063
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 109.00
Output dim: 37, lower bound: -39.2873741, upper bound: 39.3023596

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -63.2521172, 8.4618549, -63.3787231, 8.5724735, -71.8245926, 71.8405762
1: -44.4647522, 6.5426855, -44.5583267, 6.6529179, -48.1917267, 48.1693611
2: -35.2757263, 10.6073790, -35.3587074, 10.6869860, -43.3455467, 43.3489876
3: -45.9675064, 7.3740587, -46.0374146, 7.4440050, -49.9134521, 49.9117203
4: -36.0844574, 19.5473003, -36.1653061, 19.6235313, -54.3993149, 54.3883896
5: -49.1281967, 11.0277386, -49.2108421, 11.0910292, -54.6759644, 54.6979294
6: -43.4413605, 19.0063190, -43.5034065, 19.0305500, -62.4719086, 62.5097275
7: -66.7280426, -0.4639320, -66.8410873, -0.3590679, -55.5196190, 55.5265884
8: -42.6354904, 24.8659782, -42.7673416, 24.9934616, -67.6289520, 67.6333160
9: -21.5547333, 16.4169216, -21.6027794, 16.4710732, -38.0258064, 38.0196991
10: -53.0605850, 17.3538208, -53.1416588, 17.4449654, -68.3526459, 68.3379669
11: -69.5221100, -11.6205158, -69.6035614, -11.5830708, -46.6080704, 46.6568336
12: -32.5114250, 30.0045166, -32.5702705, 30.0632210, -59.6494865, 59.6529808
13: -35.7826729, 37.1974411, -35.8485756, 37.2540627, -73.0367355, 73.0460205
14: -105.1307755, -11.1360483, -105.2726440, -10.9999361, -88.8836975, 88.8863373
15: -35.1555748, 21.8265648, -35.2218246, 21.8911171, -56.6503410, 56.6363144
16: -60.8807449, 2.2158289, -60.9540787, 2.2836246, -54.7742996, 54.7770996
17: -122.7423859, -17.7653580, -122.8578568, -17.6554832, -82.2075653, 82.2277679
18: -46.8284225, 24.0820751, -46.9057083, 24.1790028, -71.0074234, 70.9877853
19: -40.1644783, 1.6995430, -40.1994553, 1.7330852, -39.6111069, 39.6203842
20: -31.6286316, 5.3635712, -31.6709518, 5.3877563, -37.0163879, 37.0345230
21: -53.1304474, 0.1561308, -53.1777344, 0.1992130, -49.7451401, 49.7527580
22: -53.8420105, 6.0436316, -53.8870544, 6.1104126, -55.6835861, 55.6652260
23: -32.7699661, 8.2510033, -32.7948380, 8.2741594, -38.6637726, 38.6967278
24: -25.9742889, 18.4462929, -25.9910965, 18.4909477, -43.8774757, 43.8502464
25: -23.3496761, 19.6630001, -23.4047909, 19.7350330, -41.5787277, 41.5638046
26: -44.7926445, 25.1875896, -44.8751450, 25.2564812, -70.0122986, 70.0158920
27: -45.6398621, 10.9273338, -45.6565018, 10.9605083, -56.6003723, 56.5838356
28: -36.1425285, 14.3615627, -36.1686058, 14.3903074, -48.7699203, 48.7964211
29: -65.1512222, -6.1978245, -65.2077942, -6.1428442, -50.0563431, 50.0588303
30: -43.7105217, 14.3893957, -43.7285538, 14.4379749, -58.1484985, 58.1179504
31: -41.5721588, 2.8285170, -41.6161652, 2.8738379, -44.4459953, 44.4446831
32: -38.6938171, 22.4112625, -38.7493057, 22.4531136, -61.1469307, 61.1605682
33: -19.5590591, 59.8693466, -19.6854649, 59.9865952, -73.2132034, 73.2285767
34: -28.1423492, 47.4354057, -28.2334442, 47.5334396, -70.2874756, 70.2790070
35: -18.5019379, 55.9111366, -18.5729828, 55.9877090, -71.3597565, 71.3390884
36: -27.1828442, 48.0951881, -27.2991199, 48.1789055, -73.4076462, 73.4417877
37: -14.6143188, 48.3423386, -14.7425346, 48.4150047, -54.6690865, 54.7305794
38: -33.1495323, 57.5026779, -33.2663651, 57.5808945, -87.2000275, 87.2407990
39: -19.5657692, 65.6082687, -19.6462631, 65.7044754, -77.0597839, 77.0553131
40: -22.7678509, 42.1833572, -22.8425713, 42.2451744, -61.5662689, 61.5749855
41: -25.9992943, 26.2100029, -26.0627556, 26.2270851, -52.2263794, 52.2727585
42: -35.5974960, 19.4937401, -35.6473427, 19.5111732, -55.1086693, 55.1410828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1141
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1164
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1163
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: A, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1156
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 1227
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1109
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1121
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1168
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 789

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2550910, upper bound: 39.1843765
time: 63.14 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2866650, upper bound: 39.2477788
time: 58.81 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -63.4074059, 8.5181122, -63.4586983, 8.5815744, -71.9889832, 71.9768066
1: -44.5783310, 6.5951996, -44.6176224, 6.6612711, -48.2917633, 48.2809830
2: -35.3578529, 10.6527309, -35.4026489, 10.6972923, -43.4099007, 43.4380913
3: -46.0034676, 7.4111633, -46.0554810, 7.4564705, -49.9476776, 49.9634132
4: -36.2140884, 19.6111774, -36.2322388, 19.6374702, -54.5363617, 54.5202599
5: -49.1982346, 11.0747061, -49.2477188, 11.1048107, -54.7308731, 54.7767639
6: -43.4866409, 19.1499996, -43.5102959, 19.1062260, -62.5928650, 62.6602936
7: -66.8397980, -0.4112911, -66.9000778, -0.3492775, -55.6035538, 55.6414185
8: -42.7866096, 24.9298248, -42.8467522, 25.0092392, -67.7958527, 67.7765808
9: -21.6253586, 16.4639854, -21.6344013, 16.4795284, -38.1048889, 38.0983887
10: -53.2382355, 17.4379902, -53.2355385, 17.4602928, -68.5302582, 68.5158691
11: -69.5995026, -11.5799503, -69.6434174, -11.5766802, -46.6567535, 46.7362328
12: -32.5593872, 30.0437317, -32.5884323, 30.0832844, -59.7316360, 59.7198524
13: -35.8484573, 37.3458519, -35.8613167, 37.3321152, -73.1805725, 73.2071686
14: -105.3951111, -11.0573273, -105.4123154, -10.9907303, -89.1305542, 89.1107025
15: -35.2896919, 21.8900528, -35.2913284, 21.9017944, -56.7866096, 56.7698097
16: -61.0130844, 2.2877216, -61.0227737, 2.2988129, -54.8826866, 54.9171028
17: -123.0166779, -17.6755867, -123.0032959, -17.6445160, -82.4091949, 82.4629593
18: -46.9431381, 24.1330566, -46.9655838, 24.1890221, -71.1321564, 71.0986404
19: -40.2286377, 1.7276697, -40.2298698, 1.7388659, -39.6544037, 39.6740913
20: -31.6724739, 5.3816166, -31.6856461, 5.3933120, -37.0657845, 37.0672607
21: -53.2200737, 0.1976480, -53.2203407, 0.2073002, -49.8188400, 49.8335304
22: -53.9737282, 6.0819073, -53.9539871, 6.1169758, -55.7917328, 55.7702370
23: -32.8159866, 8.2739639, -32.8156662, 8.2807178, -38.7156410, 38.7398682
24: -26.0339298, 18.4676170, -26.0199432, 18.4946404, -43.9407730, 43.9055290
25: -23.4007282, 19.6839066, -23.4230251, 19.7403946, -41.6380997, 41.6043892
26: -44.9006882, 25.2304039, -44.9274139, 25.2648048, -70.1285172, 70.1291885
27: -45.7045021, 10.9571075, -45.6884842, 10.9671192, -56.6716232, 56.6455917
28: -36.1784210, 14.3819199, -36.1853600, 14.3978271, -48.8228073, 48.8355217
29: -65.2909241, -6.1546326, -65.2814255, -6.1377678, -50.1584511, 50.1759834
30: -43.7670746, 14.4219294, -43.7558136, 14.4463425, -58.2134171, 58.1777420
31: -41.6124725, 2.8614807, -41.6315498, 2.8808270, -44.4933014, 44.4930305
32: -38.7580223, 22.5539474, -38.7648811, 22.5289307, -61.2869530, 61.3188286
33: -19.6615295, 60.0567932, -19.7058010, 60.0877457, -73.4161530, 73.4140930
34: -28.2118683, 47.5788002, -28.2512112, 47.6096039, -70.4329681, 70.4078674
35: -18.5780869, 56.0877838, -18.5862141, 56.0829849, -71.5297394, 71.5074539
36: -27.2681637, 48.3212814, -27.3123016, 48.3016815, -73.6161423, 73.6633224
37: -14.7070827, 48.4271927, -14.7655296, 48.4608955, -54.8094940, 54.8100090
38: -33.2624969, 57.7515030, -33.2866020, 57.7155838, -87.4465332, 87.4743271
39: -19.6732178, 65.8470078, -19.6668472, 65.8357086, -77.2985382, 77.2818680
40: -22.8382797, 42.3247948, -22.8614006, 42.3208771, -61.7112961, 61.6949844
41: -26.0504265, 26.3198967, -26.0733070, 26.2845612, -52.3349876, 52.3932037
42: -35.6420441, 19.5839615, -35.6582947, 19.5581818, -55.2002258, 55.2422562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1141
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 1140
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1164
type: B, layer: 1, pos: 1164
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1163
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: A, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 1227
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1109
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1132
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 789

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2231997, upper bound: 39.2430298
time: 78.23 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2866649, upper bound: 39.2746167
time: 195.16 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -63.2772751, 8.4714355, -63.4676514, 8.6417198, -71.9189911, 71.9390869
1: -44.4879303, 6.5490227, -44.6167221, 6.7594156, -48.3332901, 48.2410812
2: -35.2791634, 10.6174498, -35.3766556, 10.7312460, -43.4554749, 43.4491234
3: -45.9722672, 7.3871059, -46.0728226, 7.4937887, -50.0535278, 50.0712662
4: -36.1087532, 19.5601768, -36.2346573, 19.7358780, -54.5815697, 54.5110016
5: -49.1335373, 11.0436144, -49.2405624, 11.1483679, -54.8959122, 54.9287186
6: -43.4459801, 19.0606804, -43.6296043, 19.1469307, -62.5929108, 62.6902847
7: -66.7389832, -0.4521713, -66.8784256, -0.2824440, -55.6808167, 55.6523361
8: -42.6618576, 24.8797073, -42.8342514, 25.0887318, -67.7505875, 67.7139587
9: -21.5698032, 16.4232502, -21.6869545, 16.5570412, -38.1268463, 38.1102066
10: -53.1144409, 17.3674412, -53.2579956, 17.6163940, -68.5848007, 68.4669952
11: -69.5504837, -11.6156359, -69.6725159, -11.4694490, -46.8296127, 46.8004799
12: -32.5302010, 30.0108719, -32.6393700, 30.0992393, -59.8106461, 59.8373795
13: -35.7905884, 37.2535400, -36.0242310, 37.3737259, -73.1643143, 73.2777710
14: -105.2312241, -11.1314392, -105.4899826, -10.7964745, -89.1948471, 89.0893555
15: -35.1901512, 21.8358459, -35.2995491, 22.0258369, -56.8684998, 56.7653389
16: -60.9166412, 2.2304792, -61.0586166, 2.4224920, -54.9277039, 54.8677902
17: -122.8309021, -17.7517433, -123.0388718, -17.4306068, -82.4037933, 82.2768250
18: -46.8856812, 24.0880146, -47.0248795, 24.3363781, -71.2220612, 71.1128922
19: -40.1872826, 1.7065973, -40.2653770, 1.7940869, -39.6068611, 39.6013451
20: -31.6437740, 5.3659172, -31.7337456, 5.4368100, -37.0805855, 37.0996628
21: -53.1717796, 0.1600342, -53.2895851, 0.2972765, -49.8276443, 49.8128777
22: -53.9085617, 6.0479326, -54.0356903, 6.2124519, -55.6926727, 55.6479263
23: -32.7894592, 8.2548771, -32.8537025, 8.3450489, -38.7350082, 38.7373734
24: -26.0241470, 18.4487419, -26.1040268, 18.5804958, -44.0500488, 43.9926720
25: -23.3709316, 19.6655235, -23.4846706, 19.8003712, -41.6168633, 41.5964394
26: -44.8559761, 25.1922550, -45.0217667, 25.4057598, -70.2617340, 70.2140198
27: -45.6875610, 10.9308863, -45.7687378, 11.0623083, -56.7498703, 56.6996231
28: -36.1604347, 14.3656721, -36.2258606, 14.4403992, -48.8273926, 48.8431854
29: -65.2142029, -6.1933136, -65.3422089, -6.0302773, -50.1742897, 50.1260109
30: -43.7553787, 14.3940620, -43.8382187, 14.5512104, -58.3065872, 58.2322807
31: -41.5882607, 2.8341689, -41.6786919, 2.9467072, -44.5349693, 44.5128593
32: -38.7124634, 22.4595490, -38.8961792, 22.5524902, -61.2649536, 61.3557281
33: -19.5720634, 59.9229889, -19.9063663, 60.0902824, -73.3375931, 73.5172577
34: -28.1554680, 47.4648209, -28.3542595, 47.6040688, -70.4028778, 70.4695892
35: -18.5113983, 55.9576340, -18.7372570, 56.0803909, -71.4443970, 71.5382309
36: -27.1941261, 48.1588783, -27.4802208, 48.2977676, -73.5351715, 73.6917725
37: -14.6339989, 48.3647194, -14.9215126, 48.4588394, -54.6717834, 54.8865662
38: -33.1647568, 57.5766830, -33.5001373, 57.7363281, -87.3488312, 87.5369644
39: -19.5842590, 65.6746521, -19.8873024, 65.8252029, -77.1823807, 77.3587189
40: -22.7875099, 42.2263794, -22.9962082, 42.3300819, -61.6892242, 61.7947998
41: -26.0054779, 26.2488956, -26.1935501, 26.3155632, -52.3210411, 52.4424438
42: -35.6046257, 19.5314178, -35.7537651, 19.5965652, -55.2011909, 55.2851830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 1195
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1164
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1163
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1160
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1222
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1212
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1156
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1208
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1208
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1109
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1105
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1104
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 769

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 789

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2550910, upper bound: 39.1843765
time: 66.07 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3016486, upper bound: 39.2477788
time: 70.07 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -63.4322128, 8.5275764, -63.5470200, 8.6506205, -72.0828323, 72.0746002
1: -44.6017227, 6.6014338, -44.6762085, 6.7675371, -48.4332047, 48.3527908
2: -35.3606262, 10.6627131, -35.4201126, 10.7414980, -43.5198135, 43.5377350
3: -46.0084190, 7.4241533, -46.0906982, 7.5062246, -50.0876083, 50.1228981
4: -36.2385674, 19.6238384, -36.3017197, 19.7495632, -54.7186890, 54.6430893
5: -49.2036552, 11.0904417, -49.2772675, 11.1621456, -54.9503937, 55.0047379
6: -43.4912338, 19.2044659, -43.6364975, 19.2227058, -62.7139397, 62.8409653
7: -66.8500900, -0.3997440, -66.9363251, -0.2730808, -55.7651405, 55.7675781
8: -42.8131714, 24.9433289, -42.9138107, 25.1039944, -67.9171677, 67.8571396
9: -21.6403351, 16.4702606, -21.7184734, 16.5654564, -38.2057915, 38.1887360
10: -53.2921448, 17.4514542, -53.3518143, 17.6316338, -68.7624054, 68.6446228
11: -69.6279449, -11.5750866, -69.7123718, -11.4630775, -46.8783340, 46.8792992
12: -32.5779724, 30.0501747, -32.6575851, 30.1193123, -59.8926926, 59.9046822
13: -35.8562965, 37.4020157, -36.0368805, 37.4518661, -73.3081665, 73.4388962
14: -105.4956055, -11.0527267, -105.6296844, -10.7872696, -89.4417267, 89.3137512
15: -35.3243561, 21.8992538, -35.3691254, 22.0362282, -57.0046921, 56.8992271
16: -61.0490036, 2.3021622, -61.1272545, 2.4375963, -55.0361366, 55.0076103
17: -123.1052094, -17.6621304, -123.1842651, -17.4196835, -82.6054230, 82.5119553
18: -47.0003929, 24.1389675, -47.0847778, 24.3463879, -71.3467789, 71.2237473
19: -40.2514610, 1.7346864, -40.2957611, 1.7998257, -39.6500969, 39.6546631
20: -31.6875134, 5.3839827, -31.7485600, 5.4423609, -37.1298752, 37.1325417
21: -53.2614479, 0.2015629, -53.3322182, 0.3053637, -49.9015350, 49.8934822
22: -54.0401344, 6.0862312, -54.1026535, 6.2190037, -55.8005676, 55.7529869
23: -32.8356133, 8.2778215, -32.8746643, 8.3515768, -38.7868767, 38.7806511
24: -26.0837688, 18.4700394, -26.1329803, 18.5841904, -44.1133728, 44.0481339
25: -23.4210091, 19.6864204, -23.5031719, 19.8057213, -41.6762352, 41.6371651
26: -44.9640808, 25.2350464, -45.0740738, 25.4140701, -70.3781509, 70.3091202
27: -45.7522087, 10.9606104, -45.8008385, 11.0688868, -56.8210945, 56.7614479
28: -36.1964226, 14.3860083, -36.2427139, 14.4478979, -48.8802376, 48.8824272
29: -65.3536987, -6.1501770, -65.4158783, -6.0252275, -50.2761879, 50.2432442
30: -43.8121185, 14.4265795, -43.8655930, 14.5595818, -58.3717003, 58.2921715
31: -41.6285629, 2.8671107, -41.6940689, 2.9536414, -44.5822029, 44.5611801
32: -38.7764702, 22.6024780, -38.9114761, 22.6285038, -61.4049759, 61.5139542
33: -19.6744499, 60.1104355, -19.9267197, 60.1914520, -73.5404968, 73.7027893
34: -28.2248650, 47.6082382, -28.3719597, 47.6802521, -70.5484390, 70.5985565
35: -18.5875130, 56.1343079, -18.7504425, 56.1756668, -71.6143646, 71.7065811
36: -27.2793312, 48.3850098, -27.4933243, 48.4205589, -73.7435532, 73.9132843
37: -14.7267475, 48.4495811, -14.9444580, 48.5047417, -54.8121567, 54.9659119
38: -33.2775879, 57.8255310, -33.5202026, 57.8710327, -87.5955811, 87.7705231
39: -19.6916122, 65.9133759, -19.9078598, 65.9564819, -77.4211121, 77.5852737
40: -22.8578377, 42.3678970, -23.0148849, 42.4058456, -61.8346329, 61.9155197
41: -26.0566483, 26.3589134, -26.2040272, 26.3732319, -52.4298782, 52.5629425
42: -35.6492920, 19.6217976, -35.7644653, 19.6436768, -55.2929688, 55.3862610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 1195
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1164
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1163
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1160
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1222
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1212
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1156
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1208
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1109
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1105
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1104
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 769

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 789

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2550910, upper bound: 39.2111452
time: 63.06 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3016486, upper bound: 39.2746165
time: 55.06 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -63.4150429, 8.5422516, -63.4708290, 8.5828419, -71.9978867, 72.0130768
1: -44.5861549, 6.6271677, -44.6272888, 6.6673708, -48.2979736, 48.3339119
2: -35.3921432, 10.6698952, -35.4253426, 10.6984940, -43.4255791, 43.4828224
3: -46.0917206, 7.4431629, -46.1095200, 7.4596977, -50.0153275, 50.0522728
4: -36.1646500, 19.5927544, -36.2098274, 19.6338348, -54.4865799, 54.5035210
5: -49.2442169, 11.0799265, -49.2759781, 11.1027222, -54.7803764, 54.8149147
6: -43.4935760, 19.0204201, -43.5366745, 19.0504799, -62.5440559, 62.5570946
7: -66.8888855, -0.3767986, -66.9350967, -0.3407269, -55.6273575, 55.7147217
8: -42.8167191, 24.9713478, -42.8694153, 25.0112457, -67.8279648, 67.8407593
9: -21.5952930, 16.4471111, -21.6240978, 16.4823380, -38.0776291, 38.0712090
10: -53.1338959, 17.4020348, -53.1803780, 17.4622612, -68.4052429, 68.4280243
11: -69.6421051, -11.5958271, -69.6732025, -11.5636997, -46.6994781, 46.7382355
12: -32.5661621, 30.0974998, -32.5838776, 30.1135864, -59.7621880, 59.7614670
13: -35.8129730, 37.2420387, -35.8609924, 37.2720070, -73.0849762, 73.1030273
14: -105.3262863, -11.0465527, -105.3804474, -10.9776392, -88.9779816, 89.0900879
15: -35.2076416, 21.8580418, -35.2511330, 21.9072285, -56.7109604, 56.7280502
16: -60.9493103, 2.2534590, -60.9944763, 2.3008528, -54.8320045, 54.8509521
17: -122.8498001, -17.7086601, -122.9178391, -17.6379299, -82.2872925, 82.3025589
18: -46.9144630, 24.2144089, -46.9293365, 24.2525177, -71.1669769, 71.1437454
19: -40.1980515, 1.7218642, -40.2165680, 1.7392659, -39.6515541, 39.6394920
20: -31.6680336, 5.3916073, -31.6861992, 5.4034328, -37.0714645, 37.0778046
21: -53.1852875, 0.1797562, -53.2060585, 0.2109222, -49.8013229, 49.8002167
22: -53.8937988, 6.1255875, -53.9121971, 6.1603203, -55.7958832, 55.7474480
23: -32.7940865, 8.2694588, -32.8058319, 8.2832336, -38.7610245, 38.7195892
24: -26.0099564, 18.5084667, -26.0044022, 18.5256538, -43.9536209, 43.9256134
25: -23.3998222, 19.7868690, -23.4171047, 19.8064880, -41.7097397, 41.6621323
26: -44.8926620, 25.2731361, -44.9032669, 25.3054161, -70.1594391, 70.1296692
27: -45.6774597, 10.9511909, -45.6756134, 10.9721870, -56.6496468, 56.6268044
28: -36.1718903, 14.3962421, -36.1775742, 14.4068766, -48.8780403, 48.8401260
29: -65.2232208, -6.1668863, -65.2457733, -6.1313238, -50.1145401, 50.1226578
30: -43.7441063, 14.4424744, -43.7392044, 14.4648199, -58.2089272, 58.1816788
31: -41.6134262, 2.8624859, -41.6268616, 2.8848677, -44.4982948, 44.4893494
32: -38.7256165, 22.4309406, -38.7668381, 22.4671822, -61.1927986, 61.1977768
33: -19.6305885, 60.0232124, -19.7046204, 60.0756874, -73.3977280, 73.3305664
34: -28.2118759, 47.5499802, -28.2551460, 47.5975647, -70.4270020, 70.3798676
35: -18.5311508, 55.9838181, -18.5881615, 56.0335541, -71.3983078, 71.3873138
36: -27.2526855, 48.1541901, -27.3186264, 48.2121201, -73.5150757, 73.4886932
37: -14.7163086, 48.4581184, -14.7756367, 48.4827652, -54.8409538, 54.7883835
38: -33.2094383, 57.5408325, -33.2939682, 57.6021843, -87.2733154, 87.2860794
39: -19.5882740, 65.6914597, -19.6634598, 65.7568054, -77.1272125, 77.0962372
40: -22.8273354, 42.2361679, -22.8663578, 42.2738419, -61.6531143, 61.6397743
41: -26.0561466, 26.2136269, -26.1009922, 26.2388515, -52.2949982, 52.3146210
42: -35.6384926, 19.5098476, -35.6729507, 19.5276566, -55.1661491, 55.1828003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1140
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1223
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1164
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1163
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: A, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1208
type: A, layer: 1, pos: 1208
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1104
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 789

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2550910, upper bound: 39.2114180
time: 53.23 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2866650, upper bound: 39.2747999
time: 57.60 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -63.5703583, 8.5985270, -63.5507660, 8.5919428, -72.1623001, 72.1492920
1: -44.6996994, 6.6796656, -44.6865883, 6.6757126, -48.3979836, 48.4455185
2: -35.4742699, 10.7152309, -35.4692841, 10.7088156, -43.4899368, 43.5719681
3: -46.1276550, 7.4803076, -46.1275749, 7.4721594, -50.0495529, 50.1040192
4: -36.2942581, 19.6566238, -36.2767792, 19.6477566, -54.6235733, 54.6353951
5: -49.3142471, 11.1268721, -49.3128281, 11.1165047, -54.8352394, 54.8937645
6: -43.5389252, 19.1641159, -43.5435867, 19.1261444, -62.6650696, 62.7077026
7: -67.0006485, -0.3241367, -66.9941025, -0.3309517, -55.7112885, 55.8295441
8: -42.9678040, 25.0352325, -42.9488373, 25.0270119, -67.9948120, 67.9840698
9: -21.6658993, 16.4941540, -21.6557255, 16.4908142, -38.1567154, 38.1498795
10: -53.3114967, 17.4862061, -53.2742538, 17.4775620, -68.5828705, 68.6059113
11: -69.7195053, -11.5552330, -69.7130508, -11.5573015, -46.7481461, 46.8176613
12: -32.6141281, 30.1367111, -32.6020508, 30.1336670, -59.8443565, 59.8283653
13: -35.8787460, 37.3904381, -35.8737259, 37.3500595, -73.2288055, 73.2641602
14: -105.5905457, -10.9678841, -105.5201111, -10.9684401, -89.2247467, 89.3144608
15: -35.3417664, 21.9215488, -35.3206177, 21.9179134, -56.8472443, 56.8616257
16: -61.0816269, 2.3252888, -61.0631714, 2.3160048, -54.9404030, 54.9909058
17: -123.1240768, -17.6189060, -123.0633011, -17.6269531, -82.4889069, 82.5377274
18: -47.0291595, 24.2654076, -46.9892044, 24.2625542, -71.2917175, 71.2546082
19: -40.2622147, 1.7500052, -40.2470016, 1.7450285, -39.6948891, 39.6931915
20: -31.7118778, 5.4096651, -31.7008724, 5.4089966, -37.1208725, 37.1105385
21: -53.2748871, 0.2212467, -53.2486763, 0.2190208, -49.8749962, 49.8809700
22: -54.0255051, 6.1638479, -53.9791260, 6.1668882, -55.9040222, 55.8524857
23: -32.8400955, 8.2924166, -32.8266602, 8.2897873, -38.8128777, 38.7627258
24: -26.0695782, 18.5297852, -26.0332355, 18.5293465, -44.0168915, 43.9809036
25: -23.4508724, 19.8077660, -23.4353409, 19.8118515, -41.7691193, 41.7027359
26: -45.0007401, 25.3159542, -44.9555588, 25.3137474, -70.2756424, 70.2429810
27: -45.7420921, 10.9809599, -45.7075996, 10.9787941, -56.7208862, 56.6885605
28: -36.2077484, 14.4165945, -36.1942978, 14.4143944, -48.9309464, 48.8792305
29: -65.3629150, -6.1237335, -65.3194275, -6.1262722, -50.2166672, 50.2398643
30: -43.8006744, 14.4750347, -43.7664757, 14.4732075, -58.2738800, 58.2415085
31: -41.6537323, 2.8954573, -41.6422577, 2.8918533, -44.5455856, 44.5377159
32: -38.7898254, 22.5736065, -38.7824326, 22.5429955, -61.3328209, 61.3560410
33: -19.7330132, 60.2106552, -19.7249222, 60.1768570, -73.6006622, 73.5160522
34: -28.2813721, 47.6933403, -28.2729282, 47.6737213, -70.5725021, 70.5087585
35: -18.6073112, 56.1604614, -18.6014061, 56.1288567, -71.5683594, 71.5556564
36: -27.3379898, 48.3802490, -27.3318119, 48.3348808, -73.7235947, 73.7102356
37: -14.8090229, 48.5429764, -14.7986336, 48.5286713, -54.9813347, 54.8677902
38: -33.3223724, 57.7896919, -33.3142281, 57.7368813, -87.5198364, 87.5196304
39: -19.6956959, 65.9302063, -19.6840591, 65.8880157, -77.3660126, 77.3227844
40: -22.8977108, 42.3775940, -22.8851814, 42.3495636, -61.7981148, 61.7598076
41: -26.1073112, 26.3235226, -26.1115475, 26.2963257, -52.4036369, 52.4350700
42: -35.6830826, 19.6000633, -35.6838913, 19.5746555, -55.2577362, 55.2839546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1140
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1223
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1164
type: B, layer: 1, pos: 1164
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1142
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1212
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1208
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1104
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 789

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2550910, upper bound: 39.2381990
time: 68.32 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2866650, upper bound: 39.3016483
time: 66.54 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -63.4402046, 8.5518398, -63.5597420, 8.6521120, -72.0923157, 72.1115799
1: -44.6093445, 6.6334925, -44.6856842, 6.7738333, -48.4394989, 48.4056206
2: -35.3955765, 10.6799603, -35.4433060, 10.7427740, -43.5355377, 43.5829964
3: -46.0964661, 7.4562340, -46.1449356, 7.5094891, -50.1553764, 50.2118187
4: -36.1889153, 19.6056175, -36.2791748, 19.7461605, -54.6688309, 54.6261444
5: -49.2495575, 11.0958042, -49.3057022, 11.1600733, -55.0003510, 55.0456924
6: -43.4982147, 19.0748005, -43.6628876, 19.1668720, -62.6650848, 62.7376862
7: -66.8998184, -0.3650322, -66.9724274, -0.2640915, -55.7885590, 55.8404541
8: -42.8430481, 24.9850979, -42.9363251, 25.1065197, -67.9495697, 67.9214249
9: -21.6103497, 16.4534264, -21.7082653, 16.5682964, -38.1786461, 38.1616898
10: -53.1877289, 17.4156609, -53.2967110, 17.6336632, -68.6374130, 68.5570450
11: -69.6704865, -11.5909424, -69.7421494, -11.4500303, -46.9210091, 46.8819084
12: -32.5849380, 30.1038437, -32.6529999, 30.1496048, -59.9233246, 59.9458542
13: -35.8208656, 37.2981415, -36.0365982, 37.3916855, -73.2125549, 73.3347397
14: -105.4266891, -11.0419312, -105.5977478, -10.7742481, -89.2890930, 89.2930603
15: -35.2422066, 21.8673439, -35.3289108, 22.0419846, -56.9291077, 56.8568840
16: -60.9851685, 2.2680874, -61.0990219, 2.4397001, -54.9853592, 54.9416313
17: -122.9383698, -17.6950893, -123.0988388, -17.4130268, -82.4834671, 82.3515778
18: -46.9717064, 24.2203522, -47.0484810, 24.4098930, -71.3815994, 71.2688293
19: -40.2208557, 1.7289281, -40.2824707, 1.8002486, -39.6473007, 39.6204376
20: -31.6831741, 5.3939528, -31.7490101, 5.4524918, -37.1356659, 37.1429634
21: -53.2265854, 0.1836481, -53.3179245, 0.3089848, -49.8838196, 49.8603249
22: -53.9603348, 6.1298609, -54.0608521, 6.2623644, -55.8049469, 55.7301826
23: -32.8135529, 8.2733135, -32.8646774, 8.3541279, -38.8322411, 38.7602348
24: -26.0597897, 18.5109024, -26.1173153, 18.6151962, -44.1261711, 44.0680237
25: -23.4210510, 19.7893772, -23.4969807, 19.8718224, -41.7478561, 41.6947937
26: -44.9560280, 25.2778034, -45.0499306, 25.4546986, -70.4107285, 70.3277359
27: -45.7251587, 10.9547348, -45.7878265, 11.0739651, -56.7991257, 56.7425613
28: -36.1897507, 14.4003468, -36.2348251, 14.4569693, -48.9354935, 48.8869209
29: -65.2862473, -6.1623831, -65.3802032, -6.0187778, -50.2323914, 50.1897736
30: -43.7889557, 14.4471588, -43.8488998, 14.5780334, -58.3669891, 58.2960587
31: -41.6295090, 2.8681417, -41.6894150, 2.9577303, -44.5872383, 44.5575562
32: -38.7442474, 22.4792042, -38.9137306, 22.5665627, -61.3108101, 61.3929367
33: -19.6435699, 60.0768661, -19.9255257, 60.1794167, -73.5221176, 73.6192856
34: -28.2250290, 47.5793991, -28.3759956, 47.6682053, -70.5423737, 70.5704575
35: -18.5406513, 56.0303040, -18.7524433, 56.1262665, -71.4829636, 71.5864868
36: -27.2639294, 48.2178421, -27.4997559, 48.3309937, -73.6426163, 73.7387085
37: -14.7359791, 48.4805069, -14.9546070, 48.5266190, -54.8436432, 54.9443169
38: -33.2246437, 57.6148415, -33.5277596, 57.7575874, -87.4220657, 87.5821762
39: -19.6067543, 65.7578354, -19.9044952, 65.8775177, -77.2498093, 77.3995972
40: -22.8469734, 42.2791862, -23.0199680, 42.3587532, -61.7761002, 61.8595810
41: -26.0623360, 26.2524948, -26.2317772, 26.3273201, -52.3896561, 52.4842720
42: -35.6456337, 19.5475502, -35.7793427, 19.6130638, -55.2586975, 55.3268929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1140
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1223
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1164
type: A, layer: 1, pos: 1164
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1224
type: B, layer: 1, pos: 1222
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1212
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1208
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1227
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1104
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 789

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2550910, upper bound: 39.2114180
time: 57.64 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3016486, upper bound: 39.2747999
time: 57.98 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -63.5951653, 8.6079988, -63.6391106, 8.6610451, -72.2562103, 72.2471085
1: -44.7230797, 6.6859150, -44.7451859, 6.7819815, -48.5394058, 48.5173111
2: -35.4770546, 10.7252359, -35.4867401, 10.7530212, -43.5998535, 43.6715927
3: -46.1325912, 7.4933033, -46.1628189, 7.5219336, -50.1894989, 50.2634964
4: -36.3186874, 19.6692848, -36.3462219, 19.7598267, -54.8059196, 54.7582207
5: -49.3196564, 11.1426058, -49.3423767, 11.1738529, -55.0547714, 55.1217575
6: -43.5435257, 19.2185593, -43.6697731, 19.2426624, -62.7861862, 62.8883324
7: -67.0109024, -0.3125954, -67.0303192, -0.2547512, -55.8728981, 55.9556885
8: -42.9943542, 25.0487423, -43.0158730, 25.1217937, -68.1161499, 68.0646133
9: -21.6808758, 16.5004425, -21.7397823, 16.5767155, -38.2575912, 38.2402267
10: -53.3653831, 17.4996567, -53.3905029, 17.6488876, -68.8149567, 68.7346649
11: -69.7479401, -11.5504045, -69.7819977, -11.4437027, -46.9697227, 46.9607506
12: -32.6327362, 30.1431599, -32.6712112, 30.1696720, -60.0053787, 60.0131836
13: -35.8866119, 37.4465599, -36.0492783, 37.4698029, -73.3564148, 73.4958344
14: -105.6910400, -10.9632740, -105.7374496, -10.7650528, -89.5359192, 89.5174866
15: -35.3764267, 21.9307556, -35.3984909, 22.0523796, -57.0652847, 56.9908600
16: -61.1175537, 2.3397598, -61.1676178, 2.4547749, -55.0937958, 55.0814400
17: -123.2126236, -17.6054459, -123.2442551, -17.4021378, -82.6851196, 82.5866699
18: -47.0864067, 24.2713318, -47.1083603, 24.4199009, -71.5063095, 71.3796921
19: -40.2850533, 1.7570167, -40.3128815, 1.8059835, -39.6905632, 39.6737671
20: -31.7269115, 5.4120259, -31.7638264, 5.4580412, -37.1849518, 37.1758537
21: -53.3162537, 0.2251778, -53.3605728, 0.3170528, -49.9576569, 49.9409065
22: -54.0918770, 6.1681757, -54.1278305, 6.2689018, -55.9128380, 55.8352623
23: -32.8597183, 8.2962627, -32.8856468, 8.3606491, -38.8841019, 38.8034935
24: -26.1194191, 18.5322075, -26.1462765, 18.6188774, -44.1894760, 44.1234779
25: -23.4711304, 19.8102913, -23.5154934, 19.8771782, -41.8072472, 41.7355461
26: -45.0641022, 25.3205929, -45.1022263, 25.4630089, -70.5271149, 70.4228210
27: -45.7897949, 10.9844646, -45.8199425, 11.0805416, -56.8703384, 56.8044052
28: -36.2257233, 14.4206820, -36.2516937, 14.4644661, -48.9883499, 48.9261436
29: -65.4257507, -6.1192083, -65.4538879, -6.0137043, -50.3343430, 50.3070107
30: -43.8457031, 14.4796886, -43.8762550, 14.5864019, -58.4321060, 58.3559418
31: -41.6698380, 2.9010768, -41.7047806, 2.9646626, -44.6344986, 44.6058578
32: -38.8082504, 22.6221085, -38.9290276, 22.6425705, -61.4508209, 61.5511360
33: -19.7459412, 60.2642670, -19.9458542, 60.2805748, -73.7249908, 73.8047791
34: -28.2944050, 47.7227936, -28.3936806, 47.7443771, -70.6879272, 70.6994019
35: -18.6167450, 56.2069702, -18.7656212, 56.2215500, -71.6529541, 71.7548065
36: -27.3491650, 48.4440155, -27.5128345, 48.4537926, -73.8509903, 73.9601822
37: -14.8286667, 48.5653534, -14.9775333, 48.5725098, -54.9840012, 55.0236473
38: -33.3374977, 57.8636475, -33.5478172, 57.8922653, -87.6688538, 87.8157883
39: -19.7140694, 65.9965591, -19.9250526, 66.0087891, -77.4885712, 77.6261673
40: -22.9172726, 42.4207115, -23.0386581, 42.4345474, -61.9215317, 61.9803085
41: -26.1135368, 26.3625031, -26.2422523, 26.3849812, -52.4985199, 52.6047554
42: -35.6903191, 19.6379185, -35.7900543, 19.6601715, -55.3504906, 55.4279709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1140
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1223
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1164
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1224
type: B, layer: 1, pos: 1222
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1212
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1208
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1227
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 1105
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 789

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2700833, upper bound: 39.2381990
time: 59.73 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3016486, upper bound: 39.3016483
time: 59.30 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 121.11 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 121.11
Output dim: 37, lower bound: -39.2550910, upper bound: 39.1843765
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 121.11
Output dim: 37, lower bound: -39.2866650, upper bound: 39.2477788
IS_A1_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 121.11
Output dim: 37, lower bound: -39.2231997, upper bound: 39.2430298
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 121.11
Output dim: 37, lower bound: -39.2866649, upper bound: 39.2746167
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 121.11
Output dim: 37, lower bound: -39.2550910, upper bound: 39.1843765
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 121.11
Output dim: 37, lower bound: -39.3016486, upper bound: 39.2477788
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 121.11
Output dim: 37, lower bound: -39.2550910, upper bound: 39.2111452
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 121.11
Output dim: 37, lower bound: -39.3016486, upper bound: 39.2746165
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 121.11
Output dim: 37, lower bound: -39.2550910, upper bound: 39.2114180
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 121.11
Output dim: 37, lower bound: -39.2866650, upper bound: 39.2747999
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 121.11
Output dim: 37, lower bound: -39.2550910, upper bound: 39.2381990
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 121.11
Output dim: 37, lower bound: -39.2866650, upper bound: 39.3016483
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 121.11
Output dim: 37, lower bound: -39.2550910, upper bound: 39.2114180
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 121.11
Output dim: 37, lower bound: -39.3016486, upper bound: 39.2747999
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 121.11
Output dim: 37, lower bound: -39.2700833, upper bound: 39.2381990
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 121.11
Output dim: 37, lower bound: -39.3016486, upper bound: 39.3016483

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -63.2505341, 8.4610043, -63.3760529, 8.5710917, -71.8216248, 71.8370590
1: -44.4641190, 6.5420408, -44.5572433, 6.6518688, -48.1898880, 48.1649513
2: -35.2755585, 10.6055326, -35.3584099, 10.6838722, -43.3496933, 43.3419800
3: -45.9672127, 7.3725967, -46.0369377, 7.4415560, -49.9296799, 49.9023132
4: -36.0839653, 19.5463066, -36.1644821, 19.6218681, -54.3907509, 54.3834572
5: -49.1278992, 11.0268707, -49.2103271, 11.0896330, -54.6881180, 54.6875343
6: -43.4407272, 19.0033188, -43.5022697, 19.0256023, -62.4663315, 62.5055885
7: -66.7278595, -0.4647217, -66.8407364, -0.3603649, -55.5197372, 55.5222435
8: -42.6353149, 24.8628235, -42.7670212, 24.9882030, -67.6235199, 67.6298447
9: -21.5537872, 16.4159908, -21.6011829, 16.4695129, -38.0233002, 38.0171738
10: -53.0592346, 17.3525486, -53.1393738, 17.4428768, -68.3466949, 68.3302917
11: -69.5190048, -11.6211386, -69.5983734, -11.5841188, -46.6045837, 46.6522293
12: -32.5105743, 30.0038872, -32.5688591, 30.0621605, -59.6459045, 59.6440353
13: -35.7820206, 37.1945229, -35.8475037, 37.2491646, -73.0311890, 73.0420227
14: -105.1265793, -11.1374397, -105.2655640, -11.0022545, -88.8791504, 88.8790054
15: -35.1544647, 21.8258743, -35.2199707, 21.8899536, -56.6480179, 56.6324959
16: -60.8792229, 2.2148285, -60.9515381, 2.2819357, -54.7655029, 54.7699165
17: -122.7328415, -17.7662716, -122.8418121, -17.6569710, -82.2060928, 82.2228851
18: -46.8262711, 24.0814819, -46.9021378, 24.1780090, -71.0042801, 70.9836197
19: -40.1631622, 1.6992049, -40.1972733, 1.7325268, -39.5966911, 39.6128769
20: -31.6278172, 5.3630414, -31.6695919, 5.3868923, -37.0147095, 37.0326347
21: -53.1279411, 0.1556549, -53.1735306, 0.1983891, -49.7319870, 49.7436218
22: -53.8334007, 6.0433292, -53.8728409, 6.1098700, -55.6750031, 55.6526604
23: -32.7674942, 8.2505207, -32.7907791, 8.2733469, -38.6619415, 38.6934128
24: -25.9724503, 18.4459229, -25.9880943, 18.4903183, -43.8718605, 43.8473778
25: -23.3481026, 19.6626167, -23.4023075, 19.7343636, -41.5764275, 41.5606575
26: -44.7885361, 25.1871490, -44.8684464, 25.2557354, -70.0027084, 70.0250092
27: -45.6354980, 10.9268837, -45.6492920, 10.9597969, -56.5952950, 56.5761757
28: -36.1376534, 14.3612747, -36.1604271, 14.3898067, -48.7642136, 48.7879295
29: -65.1431503, -6.1980934, -65.1945724, -6.1433029, -50.0542183, 50.0540085
30: -43.7038803, 14.3886356, -43.7174072, 14.4367142, -58.1405945, 58.1060410
31: -41.5708199, 2.8279643, -41.6139526, 2.8729000, -44.4437180, 44.4419174
32: -38.6928024, 22.4100189, -38.7475891, 22.4510574, -61.1438599, 61.1576080
33: -19.5577698, 59.8685722, -19.6833839, 59.9852943, -73.2105713, 73.2180023
34: -28.1407700, 47.4350967, -28.2308998, 47.5328674, -70.2848969, 70.2693329
35: -18.5006618, 55.9107933, -18.5709801, 55.9871292, -71.3577118, 71.3339081
36: -27.1807632, 48.0946426, -27.2957268, 48.1779785, -73.4042358, 73.4371414
37: -14.6126080, 48.3419228, -14.7396927, 48.4143219, -54.6665306, 54.7091637
38: -33.1482086, 57.5011177, -33.2642059, 57.5783081, -87.1927490, 87.2250061
39: -19.5643635, 65.6074677, -19.6439934, 65.7031555, -77.0567245, 77.0509644
40: -22.7666664, 42.1811867, -22.8405952, 42.2415276, -61.5621567, 61.5705109
41: -25.9978371, 26.2092609, -26.0605202, 26.2258644, -52.2237015, 52.2697830
42: -35.5968819, 19.4925842, -35.6462746, 19.5092392, -55.1061211, 55.1388588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 1140
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1164
type: B, layer: 1, pos: 1164
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1163
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1222
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 1224
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1095
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1156
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1208
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 1105
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1121
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 791

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.1494628, upper bound: 39.0874015
time: 167.03 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2861286, upper bound: 39.2472417
time: 56.16 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -63.4047279, 8.5167055, -63.4570961, 8.5807190, -71.9854431, 71.9738007
1: -44.5772362, 6.5941267, -44.6170006, 6.6606112, -48.2873344, 48.2791328
2: -35.3575516, 10.6496267, -35.4024734, 10.6954498, -43.4029083, 43.4422569
3: -46.0029755, 7.4087181, -46.0551758, 7.4549999, -49.9382553, 49.9796295
4: -36.2132721, 19.6095161, -36.2317581, 19.6365051, -54.5313950, 54.5117264
5: -49.1977386, 11.0732985, -49.2473907, 11.1039505, -54.7204971, 54.7889175
6: -43.4855156, 19.1450462, -43.5096245, 19.1032391, -62.5887527, 62.6546707
7: -66.8394699, -0.4125938, -66.8998947, -0.3500805, -55.5992126, 55.6415024
8: -42.7862968, 24.9245510, -42.8465576, 25.0061073, -67.7924042, 67.7711105
9: -21.6237602, 16.4624290, -21.6334610, 16.4786053, -38.1023636, 38.0958900
10: -53.2359772, 17.4358997, -53.2341766, 17.4590340, -68.5225906, 68.5099106
11: -69.5942993, -11.5809946, -69.6403351, -11.5772953, -46.6521606, 46.7327194
12: -32.5579605, 30.0426521, -32.5875549, 30.0826588, -59.7226830, 59.7162704
13: -35.8473740, 37.3409653, -35.8606567, 37.3292046, -73.1765747, 73.2016220
14: -105.3880463, -11.0596733, -105.4081116, -10.9921303, -89.1231995, 89.1061249
15: -35.2878227, 21.8889084, -35.2902145, 21.9010849, -56.7827873, 56.7674675
16: -61.0105438, 2.2860050, -61.0212555, 2.2977915, -54.8754959, 54.9082870
17: -123.0006561, -17.6770229, -122.9937973, -17.6454449, -82.4043427, 82.4615250
18: -46.9396095, 24.1320553, -46.9634247, 24.1884289, -71.1280365, 71.0954819
19: -40.2264633, 1.7271237, -40.2285576, 1.7385268, -39.6468887, 39.6596985
20: -31.6711273, 5.3807511, -31.6848202, 5.3927870, -37.0639153, 37.0655708
21: -53.2158928, 0.1968327, -53.2178459, 0.2068262, -49.8097038, 49.8203697
22: -53.9595413, 6.0813465, -53.9453850, 6.1166573, -55.7791824, 55.7616653
23: -32.8119011, 8.2731495, -32.8132019, 8.2802219, -38.7123184, 38.7380295
24: -26.0309277, 18.4669952, -26.0181026, 18.4942665, -43.9379082, 43.8999176
25: -23.3982220, 19.6832314, -23.4214554, 19.7399940, -41.6349525, 41.6020927
26: -44.8939972, 25.2296600, -44.9233208, 25.2643719, -70.1376572, 70.1195679
27: -45.6972656, 10.9563780, -45.6841164, 10.9666882, -56.6639557, 56.6404953
28: -36.1702347, 14.3814278, -36.1804657, 14.3975363, -48.8142967, 48.8297844
29: -65.2777405, -6.1551132, -65.2733612, -6.1380405, -50.1536560, 50.1738472
30: -43.7559242, 14.4206591, -43.7491684, 14.4456100, -58.2015343, 58.1698265
31: -41.6102524, 2.8605571, -41.6301994, 2.8802710, -44.4905243, 44.4907570
32: -38.7563019, 22.5518913, -38.7638741, 22.5276909, -61.2839928, 61.3157654
33: -19.6594620, 60.0555000, -19.7045059, 60.0869637, -73.4056244, 73.4114761
34: -28.2093277, 47.5782089, -28.2496204, 47.6092834, -70.4233246, 70.4052734
35: -18.5761032, 56.0872345, -18.5849571, 56.0826416, -71.5245514, 71.5054321
36: -27.2647266, 48.3203583, -27.3102016, 48.3011475, -73.6115112, 73.6598969
37: -14.7042694, 48.4265137, -14.7638054, 48.4605026, -54.7880630, 54.8074417
38: -33.2603378, 57.7489090, -33.2852859, 57.7140503, -87.4307480, 87.4670486
39: -19.6709213, 65.8456726, -19.6654549, 65.8348999, -77.2941666, 77.2787933
40: -22.8363266, 42.3211670, -22.8602180, 42.3187065, -61.7068100, 61.6908836
41: -26.0482101, 26.3186779, -26.0718994, 26.2838306, -52.3320389, 52.3905792
42: -35.6409683, 19.5820332, -35.6576691, 19.5570335, -55.1980019, 55.2397003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=255, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1141
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1164
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1163
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1222
type: A, layer: 1, pos: 1224
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1212
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: A, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1227
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1109
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1121
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1168
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 791

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.1261373, upper bound: 39.1375943
time: 58.72 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2861286, upper bound: 39.2740780
time: 97.27 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -63.2756767, 8.4705839, -63.4649544, 8.6403351, -71.9160156, 71.9355392
1: -44.4873009, 6.5483780, -44.6156425, 6.7583408, -48.3314095, 48.2366486
2: -35.2790070, 10.6155968, -35.3763657, 10.7281504, -43.4596138, 43.4421310
3: -45.9719658, 7.3856583, -46.0723381, 7.4913311, -50.0697479, 50.0618591
4: -36.1082687, 19.5591621, -36.2338448, 19.7342033, -54.5729980, 54.5060844
5: -49.1332283, 11.0427418, -49.2400589, 11.1469507, -54.9080505, 54.9183273
6: -43.4453125, 19.0576973, -43.6284637, 19.1420021, -62.5873146, 62.6861610
7: -66.7387848, -0.4529419, -66.8781128, -0.2837257, -55.6809158, 55.6479988
8: -42.6616478, 24.8765488, -42.8339310, 25.0834541, -67.7451019, 67.7104797
9: -21.5688553, 16.4223156, -21.6853619, 16.5554752, -38.1243286, 38.1076775
10: -53.1130981, 17.3661690, -53.2557220, 17.6142921, -68.5788803, 68.4593048
11: -69.5473785, -11.6162796, -69.6673279, -11.4704762, -46.8261108, 46.7958984
12: -32.5293541, 30.0102215, -32.6379700, 30.0982037, -59.8070831, 59.8284225
13: -35.7899246, 37.2506180, -36.0231552, 37.3688583, -73.1587830, 73.2737732
14: -105.2270050, -11.1328373, -105.4829025, -10.7988119, -89.1903000, 89.0820084
15: -35.1890259, 21.8351517, -35.2976761, 22.0246925, -56.8661728, 56.7614975
16: -60.9151077, 2.2294550, -61.0560799, 2.4208117, -54.9188805, 54.8605957
17: -122.8214111, -17.7526379, -123.0228424, -17.4320240, -82.4023056, 82.2719650
18: -46.8835373, 24.0874081, -47.0213394, 24.3354034, -71.2189407, 71.1087494
19: -40.1859665, 1.7062588, -40.2631912, 1.7935238, -39.5924606, 39.5938339
20: -31.6429501, 5.3654051, -31.7323914, 5.4359407, -37.0788918, 37.0977974
21: -53.1692581, 0.1595478, -53.2854080, 0.2964745, -49.8145027, 49.8037415
22: -53.8999710, 6.0476446, -54.0214653, 6.2119226, -55.6841125, 55.6353645
23: -32.7869644, 8.2543840, -32.8496323, 8.3442335, -38.7331696, 38.7340622
24: -26.0223064, 18.4483700, -26.1010094, 18.5798874, -44.0444260, 43.9898148
25: -23.3693523, 19.6651230, -23.4822083, 19.7997017, -41.6145477, 41.5932846
26: -44.8519058, 25.1917973, -45.0150604, 25.4050255, -70.2569275, 70.2068558
27: -45.6831894, 10.9304399, -45.7615128, 11.0615864, -56.7447739, 56.6919518
28: -36.1555557, 14.3653784, -36.2176514, 14.4399061, -48.8216591, 48.8346863
29: -65.2061691, -6.1935892, -65.3290329, -6.0307474, -50.1721497, 50.1212234
30: -43.7487450, 14.3933058, -43.8270721, 14.5499773, -58.2987213, 58.2203789
31: -41.5869293, 2.8336124, -41.6764832, 2.9457712, -44.5326996, 44.5100937
32: -38.7114410, 22.4582939, -38.8944740, 22.5504284, -61.2618713, 61.3527679
33: -19.5707550, 59.9222374, -19.9042854, 60.0889778, -73.3349609, 73.5067291
34: -28.1538849, 47.4645004, -28.3517303, 47.6035309, -70.4002838, 70.4599457
35: -18.5101814, 55.9572945, -18.7352638, 56.0798225, -71.4423752, 71.5330353
36: -27.1920471, 48.1583214, -27.4768467, 48.2968597, -73.5317688, 73.6871338
37: -14.6323051, 48.3643188, -14.9186840, 48.4581528, -54.6692276, 54.8651466
38: -33.1634254, 57.5751038, -33.4979515, 57.7337303, -87.3415833, 87.5211487
39: -19.5828629, 65.6738434, -19.8850212, 65.8238678, -77.1793365, 77.3543396
40: -22.7863426, 42.2242241, -22.9942513, 42.3264465, -61.6850815, 61.7903366
41: -26.0040569, 26.2481632, -26.1913280, 26.3143463, -52.3184052, 52.4394913
42: -35.6039963, 19.5302696, -35.7526970, 19.5946503, -55.1986465, 55.2829666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 1140
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1195
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1164
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1163
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1160
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1222
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1212
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1156
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1208
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1109
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1218
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1104
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 769

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 791

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.1646474, upper bound: 39.1543452
time: 55.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3011099, upper bound: 39.2472417
time: 44.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -63.4306183, 8.5267296, -63.5443268, 8.6492596, -72.0798798, 72.0710602
1: -44.6010704, 6.6007843, -44.6751251, 6.7664795, -48.4313431, 48.3483772
2: -35.3604736, 10.6608629, -35.4198151, 10.7384062, -43.5239487, 43.5307236
3: -46.0081329, 7.4226885, -46.0902328, 7.5037603, -50.1038284, 50.1134987
4: -36.2380753, 19.6228523, -36.3009109, 19.7479076, -54.7101250, 54.6381645
5: -49.2033348, 11.0895729, -49.2767296, 11.1607294, -54.9625130, 54.9943581
6: -43.4905701, 19.2014923, -43.6353569, 19.2177715, -62.7083435, 62.8368492
7: -66.8498764, -0.4005413, -66.9359741, -0.2743912, -55.7652588, 55.7632446
8: -42.8129616, 24.9401817, -42.9134865, 25.0987263, -67.9116898, 67.8536682
9: -21.6393909, 16.4693336, -21.7168808, 16.5638885, -38.2032776, 38.1862144
10: -53.2907982, 17.4501724, -53.3495445, 17.6295280, -68.7564392, 68.6369476
11: -69.6248398, -11.5757246, -69.7071609, -11.4641285, -46.8748131, 46.8747025
12: -32.5771332, 30.0495377, -32.6561584, 30.1182671, -59.8890839, 59.8957329
13: -35.8556557, 37.3990746, -36.0358238, 37.4469833, -73.3026428, 73.4348984
14: -105.4914246, -11.0541410, -105.6226349, -10.7896242, -89.4371948, 89.3064575
15: -35.3232422, 21.8985653, -35.3672638, 22.0350685, -57.0023422, 56.8954124
16: -61.0474892, 2.3011570, -61.1247139, 2.4358635, -55.0273438, 55.0004234
17: -123.0956879, -17.6630535, -123.1682205, -17.4211121, -82.6039429, 82.5070724
18: -46.9982529, 24.1383591, -47.0812073, 24.3454094, -71.3436584, 71.2195663
19: -40.2501488, 1.7343521, -40.2935867, 1.7992516, -39.6357155, 39.6471481
20: -31.6866913, 5.3834610, -31.7472153, 5.4415078, -37.1282005, 37.1306763
21: -53.2589340, 0.2010803, -53.3280220, 0.3045578, -49.8883591, 49.8843231
22: -54.0315399, 6.0859232, -54.0884628, 6.2184515, -55.7920227, 55.7404327
23: -32.8331337, 8.2773237, -32.8705978, 8.3507624, -38.7850342, 38.7773094
24: -26.0819225, 18.4696655, -26.1299686, 18.5835686, -44.1077385, 44.0452576
25: -23.4194317, 19.6860085, -23.5006886, 19.8050499, -41.6739311, 41.6340218
26: -44.9599915, 25.2345810, -45.0673676, 25.4133434, -70.3733368, 70.3019485
27: -45.7478104, 10.9601707, -45.7936134, 11.0681591, -56.8159714, 56.7537842
28: -36.1915283, 14.3857079, -36.2345200, 14.4473972, -48.8745270, 48.8738976
29: -65.3456650, -6.1504507, -65.4026947, -6.0256929, -50.2740860, 50.2384415
30: -43.8054771, 14.4258423, -43.8544502, 14.5583229, -58.3638000, 58.2802925
31: -41.6272278, 2.8665571, -41.6918488, 2.9527178, -44.5799446, 44.5584068
32: -38.7754517, 22.6012268, -38.9097824, 22.6264381, -61.4018898, 61.5110092
33: -19.6731510, 60.1096878, -19.9246330, 60.1901703, -73.5378876, 73.6922684
34: -28.2232857, 47.6078796, -28.3694096, 47.6796799, -70.5458374, 70.5889130
35: -18.5862732, 56.1339722, -18.7484665, 56.1751137, -71.6123352, 71.7013855
36: -27.2772522, 48.3844833, -27.4899063, 48.4196358, -73.7401505, 73.9086533
37: -14.7250290, 48.4491768, -14.9416313, 48.5040550, -54.8095818, 54.9444923
38: -33.2762680, 57.8239899, -33.5180435, 57.8684044, -87.5883789, 87.7547684
39: -19.6901703, 65.9125900, -19.9056015, 65.9551392, -77.4180832, 77.5809021
40: -22.8566628, 42.3657379, -23.0129089, 42.4021988, -61.8305054, 61.9110413
41: -26.0552120, 26.3581734, -26.2018051, 26.3720055, -52.4272156, 52.5599785
42: -35.6486359, 19.6206455, -35.7634125, 19.6417503, -55.2903862, 55.3840561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1195
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1223
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1164
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1163
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1160
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1222
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1212
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1095
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1109
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1105
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1218
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1104
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1120
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 769

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 791

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.1646474, upper bound: 39.1812813
time: 63.12 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3011099, upper bound: 39.2740781
time: 58.46 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -63.4134598, 8.5413914, -63.4681358, 8.5814562, -71.9949188, 72.0095291
1: -44.5855103, 6.6265182, -44.6262207, 6.6662884, -48.2961121, 48.3294945
2: -35.3919754, 10.6680517, -35.4250641, 10.6953840, -43.4297256, 43.4758530
3: -46.0914154, 7.4417057, -46.1090317, 7.4572592, -50.0315552, 50.0428886
4: -36.1641541, 19.5917702, -36.2090111, 19.6321697, -54.4780121, 54.4985886
5: -49.2439041, 11.0790653, -49.2754440, 11.1013193, -54.7925415, 54.8045197
6: -43.4929123, 19.0174580, -43.5355492, 19.0455494, -62.5384598, 62.5530090
7: -66.8886719, -0.3775845, -66.9347534, -0.3420544, -55.6274529, 55.7103996
8: -42.8165283, 24.9681854, -42.8690948, 25.0059471, -67.8224792, 67.8372803
9: -21.5943432, 16.4461708, -21.6225014, 16.4807892, -38.0751343, 38.0686722
10: -53.1325378, 17.4007759, -53.1781311, 17.4601517, -68.3993225, 68.4203339
11: -69.6390076, -11.5964413, -69.6680145, -11.5647364, -46.6959724, 46.7336388
12: -32.5653000, 30.0968533, -32.5824814, 30.1125126, -59.7585716, 59.7525101
13: -35.8123093, 37.2391167, -35.8599129, 37.2671509, -73.0794601, 73.0990295
14: -105.3220749, -11.0479422, -105.3733826, -10.9799480, -88.9733429, 89.0827789
15: -35.2065125, 21.8573513, -35.2492447, 21.9060822, -56.7086067, 56.7242355
16: -60.9477921, 2.2524309, -60.9919472, 2.2991657, -54.8231888, 54.8437500
17: -122.8402557, -17.7095261, -122.9018021, -17.6394005, -82.2858047, 82.2976761
18: -46.9123230, 24.2138195, -46.9258041, 24.2515202, -71.1638412, 71.1396255
19: -40.1967316, 1.7215281, -40.2144051, 1.7387033, -39.6371651, 39.6319771
20: -31.6671982, 5.3910813, -31.6848469, 5.4025698, -37.0697670, 37.0759277
21: -53.1827583, 0.1792545, -53.2018890, 0.2101078, -49.7881546, 49.7910995
22: -53.8852158, 6.1252537, -53.8980179, 6.1598148, -55.7872925, 55.7348900
23: -32.7916031, 8.2689629, -32.8017731, 8.2824059, -38.7592049, 38.7162552
24: -26.0081139, 18.5081024, -26.0014095, 18.5250225, -43.9479675, 43.9227295
25: -23.3982296, 19.7864552, -23.4146194, 19.8058243, -41.7074051, 41.6589699
26: -44.8886032, 25.2726746, -44.8966064, 25.3046570, -70.1498108, 70.1387787
27: -45.6730728, 10.9507484, -45.6683960, 10.9714699, -56.6445427, 56.6191444
28: -36.1669922, 14.3959579, -36.1693840, 14.4063969, -48.8723068, 48.8316040
29: -65.2151566, -6.1671648, -65.2325897, -6.1317959, -50.1124077, 50.1178551
30: -43.7374573, 14.4417305, -43.7280769, 14.4635677, -58.2010269, 58.1698074
31: -41.6120834, 2.8619270, -41.6246567, 2.8839431, -44.4960251, 44.4865837
32: -38.7245827, 22.4296913, -38.7651443, 22.4651337, -61.1897163, 61.1948357
33: -19.6293087, 60.0224457, -19.7025528, 60.0743942, -73.3951111, 73.3200302
34: -28.2103043, 47.5496292, -28.2526073, 47.5969925, -70.4243774, 70.3702087
35: -18.5298901, 55.9834938, -18.5861683, 56.0330162, -71.3962555, 71.3821411
36: -27.2506027, 48.1536217, -27.3152046, 48.2112045, -73.5116882, 73.4840546
37: -14.7145844, 48.4577293, -14.7728062, 48.4820786, -54.8383598, 54.7669296
38: -33.2080879, 57.5392876, -33.2917938, 57.5995903, -87.2660217, 87.2702560
39: -19.5868835, 65.6906509, -19.6611786, 65.7554779, -77.1241684, 77.0918884
40: -22.8261662, 42.2340012, -22.8643856, 42.2701950, -61.6490097, 61.6353226
41: -26.0547180, 26.2128963, -26.0987453, 26.2376328, -52.2923508, 52.3116417
42: -35.6378555, 19.5087090, -35.6718750, 19.5257244, -55.1635818, 55.1805840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1140
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1223
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1164
type: B, layer: 1, pos: 1164
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1208
type: A, layer: 1, pos: 1208
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 1105
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 791

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.1494628, upper bound: 39.1813890
time: 63.48 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2861286, upper bound: 39.2742629
time: 66.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -63.5687561, 8.5976515, -63.5481071, 8.5905619, -72.1593170, 72.1457596
1: -44.6990547, 6.6790361, -44.6855011, 6.6746264, -48.3961105, 48.4410973
2: -35.4740982, 10.7134047, -35.4689941, 10.7057037, -43.4940910, 43.5649643
3: -46.1273575, 7.4788284, -46.1270905, 7.4696913, -50.0657883, 50.0946045
4: -36.2937660, 19.6556110, -36.2759705, 19.6460991, -54.6150131, 54.6304550
5: -49.3139229, 11.1260090, -49.3122978, 11.1150875, -54.8474083, 54.8833809
6: -43.5382462, 19.1611214, -43.5424538, 19.1212044, -62.6594505, 62.7035751
7: -67.0004349, -0.3249378, -66.9937592, -0.3322697, -55.7113571, 55.8252182
8: -42.9675980, 25.0320930, -42.9485168, 25.0217400, -67.9893341, 67.9806061
9: -21.6649513, 16.4932117, -21.6541328, 16.4892578, -38.1542091, 38.1473465
10: -53.3101387, 17.4849358, -53.2719803, 17.4754944, -68.5768890, 68.5982361
11: -69.7164307, -11.5558567, -69.7078857, -11.5583439, -46.7446556, 46.8130684
12: -32.6133041, 30.1360512, -32.6006355, 30.1326370, -59.8407669, 59.8194046
13: -35.8780746, 37.3874893, -35.8726616, 37.3451996, -73.2232742, 73.2601471
14: -105.5863419, -10.9692373, -105.5130463, -10.9707718, -89.2201385, 89.3071289
15: -35.3406296, 21.9208584, -35.3187408, 21.9167633, -56.8449097, 56.8577843
16: -61.0801086, 2.3242569, -61.0606346, 2.3143024, -54.9316254, 54.9837074
17: -123.1145401, -17.6198063, -123.0472946, -17.6283913, -82.4874420, 82.5328827
18: -47.0270348, 24.2647972, -46.9856491, 24.2615623, -71.2885971, 71.2504425
19: -40.2609062, 1.7496696, -40.2448044, 1.7444839, -39.6804962, 39.6856766
20: -31.7110519, 5.4091349, -31.6995182, 5.4081240, -37.1191750, 37.1086540
21: -53.2724113, 0.2207842, -53.2445030, 0.2181969, -49.8618546, 49.8718681
22: -54.0169144, 6.1635313, -53.9649391, 6.1663532, -55.8954506, 55.8399086
23: -32.8376122, 8.2919273, -32.8225937, 8.2889671, -38.8110466, 38.7593994
24: -26.0677395, 18.5294189, -26.0302410, 18.5287189, -44.0112724, 43.9780273
25: -23.4492931, 19.8073616, -23.4328556, 19.8111763, -41.7668076, 41.6995850
26: -44.9966660, 25.3154964, -44.9488831, 25.3130074, -70.2659760, 70.2521057
27: -45.7377167, 10.9805183, -45.7003860, 10.9780731, -56.7157898, 56.6809044
28: -36.2028694, 14.4162998, -36.1861191, 14.4139175, -48.9252205, 48.8707275
29: -65.3548889, -6.1239958, -65.3062592, -6.1267014, -50.2145386, 50.2350349
30: -43.7940407, 14.4742699, -43.7553406, 14.4719505, -58.2659912, 58.2296104
31: -41.6523972, 2.8948932, -41.6400490, 2.8909249, -44.5433235, 44.5349426
32: -38.7887878, 22.5723743, -38.7807159, 22.5409317, -61.3297195, 61.3530884
33: -19.7317200, 60.2098846, -19.7228737, 60.1755600, -73.5980453, 73.5055161
34: -28.2798023, 47.6930084, -28.2703743, 47.6731644, -70.5699310, 70.4991150
35: -18.6060638, 56.1601295, -18.5994301, 56.1283035, -71.5663147, 71.5504608
36: -27.3359032, 48.3797188, -27.3283710, 48.3339920, -73.7201843, 73.7056122
37: -14.8073082, 48.5425606, -14.7957649, 48.5279961, -54.9787521, 54.8463745
38: -33.3210297, 57.7880936, -33.3120575, 57.7342720, -87.5125885, 87.5038147
39: -19.6942787, 65.9293976, -19.6817703, 65.8866806, -77.3629303, 77.3184204
40: -22.8965340, 42.3754272, -22.8832207, 42.3459206, -61.7940063, 61.7553368
41: -26.1058960, 26.3227749, -26.1093178, 26.2950993, -52.4009933, 52.4320908
42: -35.6824493, 19.5989342, -35.6828156, 19.5727310, -55.2551804, 55.2817497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1140
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1223
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1164
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1142
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1212
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1095
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1208
type: A, layer: 1, pos: 1208
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1105
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1127
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 791

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.1494628, upper bound: 39.2083301
time: 53.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.1143869, upper bound: 39.1413468
time: 118.39 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -63.4386215, 8.5509977, -63.5570412, 8.6507139, -72.0893326, 72.1080399
1: -44.6087112, 6.6328678, -44.6846085, 6.7727766, -48.4376335, 48.4011841
2: -35.3954086, 10.6781178, -35.4430237, 10.7396536, -43.5396576, 43.5759888
3: -46.0961685, 7.4547720, -46.1444359, 7.5070429, -50.1716156, 50.2024269
4: -36.1884155, 19.6046124, -36.2783813, 19.7445145, -54.6602516, 54.6212082
5: -49.2492332, 11.0949564, -49.3051605, 11.1586704, -55.0124550, 55.0353088
6: -43.4975433, 19.0718269, -43.6617355, 19.1619453, -62.6594887, 62.7335625
7: -66.8996124, -0.3658104, -66.9721146, -0.2653866, -55.7886276, 55.8361320
8: -42.8428535, 24.9819412, -42.9360275, 25.1012440, -67.9440994, 67.9179688
9: -21.6094017, 16.4524956, -21.7066727, 16.5667439, -38.1761475, 38.1591682
10: -53.1863708, 17.4143810, -53.2944527, 17.6315804, -68.6314697, 68.5493774
11: -69.6673965, -11.5915670, -69.7369537, -11.4510918, -46.9175072, 46.8773003
12: -32.5841064, 30.1031952, -32.6516037, 30.1485710, -59.9197426, 59.9369202
13: -35.8202095, 37.2952042, -36.0355339, 37.3868065, -73.2070160, 73.3307343
14: -105.4224701, -11.0433426, -105.5907059, -10.7765560, -89.2845001, 89.2857513
15: -35.2410698, 21.8666382, -35.3270531, 22.0408325, -56.9267578, 56.8530464
16: -60.9836617, 2.2670507, -61.0964622, 2.4380150, -54.9765396, 54.9344368
17: -122.9288101, -17.6959629, -123.0827942, -17.4144878, -82.4819641, 82.3467560
18: -46.9695740, 24.2197628, -47.0449219, 24.4089184, -71.3784943, 71.2646866
19: -40.2195435, 1.7285829, -40.2803078, 1.7996874, -39.6329193, 39.6129265
20: -31.6823483, 5.3934298, -31.7476578, 5.4516230, -37.1339722, 37.1410866
21: -53.2240829, 0.1831436, -53.3137398, 0.3082018, -49.8706589, 49.8512001
22: -53.9517441, 6.1295595, -54.0466347, 6.2618284, -55.7963562, 55.7176170
23: -32.8110771, 8.2728291, -32.8606339, 8.3533154, -38.8304138, 38.7569084
24: -26.0579529, 18.5105438, -26.1142979, 18.6145725, -44.1205521, 44.0651665
25: -23.4194565, 19.7889748, -23.4945164, 19.8711529, -41.7455368, 41.6916313
26: -44.9519348, 25.2773399, -45.0432434, 25.4539433, -70.4058762, 70.3205872
27: -45.7207756, 10.9542847, -45.7806053, 11.0732508, -56.7940254, 56.7348900
28: -36.1848679, 14.4000492, -36.2266464, 14.4564781, -48.9297562, 48.8783951
29: -65.2782135, -6.1626472, -65.3670425, -6.0192175, -50.2302513, 50.1849556
30: -43.7823181, 14.4463825, -43.8377380, 14.5767937, -58.3591118, 58.2841187
31: -41.6281776, 2.8675842, -41.6872101, 2.9567971, -44.5849762, 44.5547943
32: -38.7432251, 22.4779682, -38.9120255, 22.5645027, -61.3077278, 61.3899918
33: -19.6422710, 60.0761108, -19.9234352, 60.1781120, -73.5195007, 73.6087341
34: -28.2234383, 47.5790520, -28.3734322, 47.6676254, -70.5397949, 70.5608063
35: -18.5393887, 56.0299873, -18.7504349, 56.1257057, -71.4809418, 71.5812912
36: -27.2618790, 48.2173080, -27.4963570, 48.3300858, -73.6392212, 73.7340698
37: -14.7342339, 48.4801025, -14.9517860, 48.5259323, -54.8410606, 54.9228821
38: -33.2233200, 57.6132660, -33.5255623, 57.7550163, -87.4147720, 87.5663452
39: -19.6053505, 65.7570496, -19.9022121, 65.8761826, -77.2467804, 77.3952560
40: -22.8458061, 42.2770157, -23.0180130, 42.3551254, -61.7719727, 61.8551140
41: -26.0609169, 26.2517662, -26.2295494, 26.3261166, -52.3870316, 52.4813156
42: -35.6449928, 19.5463867, -35.7782784, 19.6111488, -55.2561417, 55.3246651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1140
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1223
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1164
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1163
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1224
type: B, layer: 1, pos: 1222
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1208
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1227
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1109
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 791

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.1646474, upper bound: 39.1813890
time: 69.27 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3011099, upper bound: 39.2742629
time: 56.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -63.5935707, 8.6071310, -63.6364212, 8.6596613, -72.2532349, 72.2435532
1: -44.7224350, 6.6852837, -44.7440987, 6.7809105, -48.5375595, 48.5128784
2: -35.4768753, 10.7233858, -35.4864578, 10.7499294, -43.6039963, 43.6646004
3: -46.1322975, 7.4918327, -46.1623344, 7.5194817, -50.2057343, 50.2540894
4: -36.3181953, 19.6682930, -36.3454132, 19.7581749, -54.7973633, 54.7532921
5: -49.3193207, 11.1417618, -49.3418579, 11.1724358, -55.0668716, 55.1113815
6: -43.5428314, 19.2156067, -43.6686325, 19.2377071, -62.7805405, 62.8842392
7: -67.0107117, -0.3133984, -67.0299988, -0.2560749, -55.8729553, 55.9513474
8: -42.9941521, 25.0455933, -43.0155563, 25.1165161, -68.1106720, 68.0611496
9: -21.6799240, 16.4995022, -21.7381859, 16.5751534, -38.2550774, 38.2376862
10: -53.3640251, 17.4983902, -53.3882332, 17.6468067, -68.8090134, 68.7269974
11: -69.7448654, -11.5510206, -69.7768097, -11.4447470, -46.9662209, 46.9561424
12: -32.6318970, 30.1425209, -32.6697922, 30.1686249, -60.0017624, 60.0042419
13: -35.8859444, 37.4436150, -36.0482063, 37.4649544, -73.3508987, 73.4918213
14: -105.6868210, -10.9646416, -105.7304001, -10.7673655, -89.5313110, 89.5101624
15: -35.3753052, 21.9300518, -35.3966141, 22.0512123, -57.0629654, 56.9870415
16: -61.1160316, 2.3387356, -61.1650925, 2.4530792, -55.0849838, 55.0742455
17: -123.2031097, -17.6063480, -123.2282181, -17.4035492, -82.6836548, 82.5818176
18: -47.0842705, 24.2707233, -47.1047974, 24.4189205, -71.5031891, 71.3755188
19: -40.2837219, 1.7566605, -40.3107033, 1.8054347, -39.6761742, 39.6662560
20: -31.7260818, 5.4115009, -31.7624645, 5.4571781, -37.1832581, 37.1739655
21: -53.3137398, 0.2246780, -53.3563805, 0.3162432, -49.9445229, 49.9317703
22: -54.0833015, 6.1678667, -54.1136436, 6.2683735, -55.9042587, 55.8227272
23: -32.8572350, 8.2957850, -32.8815842, 8.3598309, -38.8822594, 38.8001709
24: -26.1175823, 18.5318279, -26.1432686, 18.6182671, -44.1838570, 44.1206055
25: -23.4695530, 19.8098869, -23.5130234, 19.8765049, -41.8049240, 41.7323990
26: -45.0600204, 25.3201351, -45.0955658, 25.4622478, -70.5222702, 70.4157028
27: -45.7854156, 10.9840298, -45.8127098, 11.0798187, -56.8652344, 56.7967377
28: -36.2208481, 14.4203930, -36.2434921, 14.4639759, -48.9826355, 48.9176140
29: -65.4177246, -6.1194811, -65.4407654, -6.0141697, -50.3322105, 50.3022156
30: -43.8390427, 14.4789352, -43.8651314, 14.5851631, -58.4242058, 58.3440666
31: -41.6684914, 2.9005189, -41.7025833, 2.9637341, -44.6322250, 44.6031036
32: -38.8072357, 22.6208839, -38.9273186, 22.6404953, -61.4477310, 61.5482025
33: -19.7446480, 60.2635231, -19.9437847, 60.2792816, -73.7223663, 73.7942505
34: -28.2928009, 47.7224388, -28.3911285, 47.7437820, -70.6853561, 70.6897736
35: -18.6155090, 56.2066422, -18.7636681, 56.2209969, -71.6508789, 71.7496185
36: -27.3470840, 48.4434433, -27.5094204, 48.4528580, -73.8475876, 73.9555511
37: -14.8269424, 48.5649605, -14.9747124, 48.5718307, -54.9814224, 55.0022316
38: -33.3361664, 57.8621140, -33.5456772, 57.8896790, -87.6616211, 87.7999802
39: -19.7126637, 65.9957886, -19.9227791, 66.0074463, -77.4855194, 77.6218262
40: -22.9160881, 42.4185333, -23.0366745, 42.4308929, -61.9173889, 61.9758301
41: -26.1121101, 26.3617630, -26.2400322, 26.3837547, -52.4958649, 52.6017952
42: -35.6896820, 19.6367741, -35.7889786, 19.6582413, -55.3479233, 55.4257507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 791
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1141
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1140
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1223
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1164
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1163
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1224
type: B, layer: 1, pos: 1222
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1208
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1227
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1109
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 1105
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 791

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.1646474, upper bound: 39.2083301
time: 55.31 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.1296010, upper bound: 39.2083301
time: 63.55 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 120.91 seconds
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 120.91
Output dim: 37, lower bound: -39.1494628, upper bound: 39.0874015
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 120.91
Output dim: 37, lower bound: -39.2861286, upper bound: 39.2472417
IS_A1_B1_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 120.91
Output dim: 37, lower bound: -39.1261373, upper bound: 39.1375943
IS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 120.91
Output dim: 37, lower bound: -39.2861286, upper bound: 39.2740780
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 120.91
Output dim: 37, lower bound: -39.1646474, upper bound: 39.1543452
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 120.91
Output dim: 37, lower bound: -39.3011099, upper bound: 39.2472417
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 120.91
Output dim: 37, lower bound: -39.1646474, upper bound: 39.1812813
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 120.91
Output dim: 37, lower bound: -39.3011099, upper bound: 39.2740781
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 120.91
Output dim: 37, lower bound: -39.1494628, upper bound: 39.1813890
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 120.91
Output dim: 37, lower bound: -39.2861286, upper bound: 39.2742629
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 120.91
Output dim: 37, lower bound: -39.1494628, upper bound: 39.2083301
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 120.91
Output dim: 37, lower bound: -39.1143869, upper bound: 39.1413468
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 120.91
Output dim: 37, lower bound: -39.1646474, upper bound: 39.1813890
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 120.91
Output dim: 37, lower bound: -39.3011099, upper bound: 39.2742629
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 120.91
Output dim: 37, lower bound: -39.1646474, upper bound: 39.2083301
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 120.91
Output dim: 37, lower bound: -39.1296010, upper bound: 39.2083301

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -63.2495270, 8.4580584, -63.3755302, 8.5694656, -71.8189926, 71.8335876
1: -44.4638977, 6.5407276, -44.5571365, 6.6511822, -48.1890182, 48.1549568
2: -35.2753868, 10.6042757, -35.3583298, 10.6832123, -43.3487778, 43.3264160
3: -45.9670181, 7.3701754, -46.0368195, 7.4399452, -49.9286957, 49.8778114
4: -36.0836296, 19.5448704, -36.1643181, 19.6211281, -54.3896484, 54.3648376
5: -49.1277046, 11.0252562, -49.2102127, 11.0887947, -54.6865463, 54.6666756
6: -43.4399834, 19.0009499, -43.5018883, 19.0243855, -62.4643707, 62.5028381
7: -66.7277222, -0.4669571, -66.8406677, -0.3616447, -55.5189552, 55.4782715
8: -42.6350250, 24.8566093, -42.7668839, 24.9849548, -67.6199799, 67.6234894
9: -21.5527668, 16.4148617, -21.6006622, 16.4688873, -38.0216522, 38.0155258
10: -53.0585976, 17.3484516, -53.1390495, 17.4407539, -68.3435898, 68.3254929
11: -69.5182953, -11.6222582, -69.5979996, -11.5847912, -46.5957031, 46.6355629
12: -32.5095825, 30.0032673, -32.5683289, 30.0618439, -59.6264725, 59.6427879
13: -35.7816315, 37.1925125, -35.8472977, 37.2481308, -73.0297623, 73.0398102
14: -105.1250916, -11.1383896, -105.2648315, -11.0027580, -88.8760223, 88.8769836
15: -35.1535301, 21.8233109, -35.2194862, 21.8885727, -56.6462097, 56.6303215
16: -60.8782692, 2.2131720, -60.9510536, 2.2809448, -54.7625427, 54.7806206
17: -122.7268143, -17.7695122, -122.8387527, -17.6586342, -82.1988297, 82.2367859
18: -46.8229370, 24.0810165, -46.9004059, 24.1777439, -71.0006790, 70.9814224
19: -40.1617622, 1.6987777, -40.1965294, 1.7323127, -39.5908279, 39.5991287
20: -31.6267872, 5.3625331, -31.6690521, 5.3866262, -37.0134125, 37.0315857
21: -53.1255226, 0.1550903, -53.1722908, 0.1981192, -49.7255745, 49.7373390
22: -53.8271828, 6.0429487, -53.8696518, 6.1096783, -55.6700783, 55.6499596
23: -32.7660828, 8.2501869, -32.7900314, 8.2731838, -38.6594467, 38.6922646
24: -25.9702835, 18.4456615, -25.9869976, 18.4901848, -43.8622246, 43.8447227
25: -23.3462791, 19.6621590, -23.4013481, 19.7341270, -41.5730057, 41.5594521
26: -44.7856445, 25.1868439, -44.8668823, 25.2555714, -69.9922409, 70.0220032
27: -45.6328278, 10.9266043, -45.6479263, 10.9596548, -56.5924835, 56.5745316
28: -36.1343575, 14.3610106, -36.1586533, 14.3896637, -48.7608910, 48.7861214
29: -65.1365356, -6.1984415, -65.1910706, -6.1434622, -50.0509033, 50.0568199
30: -43.7019997, 14.3880310, -43.7164650, 14.4363918, -58.1383896, 58.1044960
31: -41.5698166, 2.8273659, -41.6134109, 2.8725977, -44.4424133, 44.4407768
32: -38.6919594, 22.4081173, -38.7471657, 22.4500656, -61.1420250, 61.1552811
33: -19.5563469, 59.8679543, -19.6826439, 59.9849663, -73.1961899, 73.2164917
34: -28.1391582, 47.4348450, -28.2300987, 47.5327606, -70.2592392, 70.2682037
35: -18.4978523, 55.9105530, -18.5694618, 55.9870033, -71.3645782, 71.3281479
36: -27.1773796, 48.0942688, -27.2939777, 48.1778030, -73.4075775, 73.4342575
37: -14.6101055, 48.3417587, -14.7384224, 48.4142227, -54.6441307, 54.7076416
38: -33.1462936, 57.4992714, -33.2632294, 57.5773621, -87.2108765, 87.2178116
39: -19.5627174, 65.6060028, -19.6430950, 65.7023926, -77.0668411, 77.0467911
40: -22.7649269, 42.1807060, -22.8397026, 42.2412720, -61.5523796, 61.5690804
41: -25.9953232, 26.2089252, -26.0592117, 26.2256870, -52.2210083, 52.2681351
42: -35.5961723, 19.4916954, -35.6459045, 19.5087872, -55.1049576, 55.1375999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=255, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1164
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 791
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1196
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1163
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1222
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1212
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: A, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1092
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 1227
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1109
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1105
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1386
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1168
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 616

## Relational analysis of IS_A1_B1_A1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2849094, upper bound: 39.2432975
time: 48.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.1103776, upper bound: 39.2432992
time: 57.48 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -63.4042244, 8.5150795, -63.4561157, 8.5777874, -71.9820099, 71.9711914
1: -44.5771294, 6.5934391, -44.6167793, 6.6593227, -48.2773514, 48.2782516
2: -35.3574600, 10.6489725, -35.4022865, 10.6941814, -43.3873405, 43.4413338
3: -46.0028839, 7.4071112, -46.0550003, 7.4525633, -49.9137611, 49.9786682
4: -36.2131119, 19.6087723, -36.2314224, 19.6350708, -54.5127983, 54.5105972
5: -49.1976242, 11.0724545, -49.2471771, 11.1023312, -54.6995811, 54.7873306
6: -43.4851608, 19.1438179, -43.5089188, 19.1008663, -62.5860291, 62.6527367
7: -66.8394165, -0.4138508, -66.8997650, -0.3522930, -55.5552559, 55.6407242
8: -42.7861633, 24.9212914, -42.8463058, 24.9998970, -67.7860565, 67.7675934
9: -21.6232338, 16.4617996, -21.6324387, 16.4774761, -38.1007080, 38.0942383
10: -53.2356339, 17.4337826, -53.2335281, 17.4549179, -68.5177765, 68.5068054
11: -69.5939178, -11.5816555, -69.6396332, -11.5784311, -46.6355209, 46.7238579
12: -32.5574455, 30.0423412, -32.5865555, 30.0820465, -59.7214584, 59.6968346
13: -35.8471642, 37.3399200, -35.8602600, 37.3271751, -73.1743393, 73.2001801
14: -105.3872833, -11.0601606, -105.4066162, -10.9930840, -89.1212082, 89.1029739
15: -35.2873497, 21.8875237, -35.2892685, 21.8985310, -56.7806396, 56.7656555
16: -61.0100479, 2.2850170, -61.0202980, 2.2961197, -54.8862076, 54.9053383
17: -122.9975891, -17.6787300, -122.9877853, -17.6486778, -82.4182281, 82.4542313
18: -46.9378548, 24.1318188, -46.9600830, 24.1879654, -71.1258240, 71.0919037
19: -40.2257233, 1.7269044, -40.2271538, 1.7381196, -39.6331558, 39.6538506
20: -31.6705856, 5.3804855, -31.6838093, 5.3922811, -37.0628662, 37.0642929
21: -53.2146378, 0.1965647, -53.2154465, 0.2062674, -49.8033943, 49.8139610
22: -53.9563141, 6.0811615, -53.9391327, 6.1162634, -55.7765045, 55.7567406
23: -32.8111725, 8.2729864, -32.8117905, 8.2798948, -38.7111549, 38.7355347
24: -26.0297928, 18.4668598, -26.0159378, 18.4940033, -43.9352417, 43.8902893
25: -23.3972759, 19.6829948, -23.4196224, 19.7395439, -41.6337700, 41.5986710
26: -44.8924637, 25.2295189, -44.9204102, 25.2640629, -70.1346054, 70.1091080
27: -45.6959076, 10.9562340, -45.6814575, 10.9664040, -56.6623116, 56.6376915
28: -36.1684570, 14.3812885, -36.1771851, 14.3972521, -48.8125229, 48.8264923
29: -65.2742615, -6.1552715, -65.2667236, -6.1383667, -50.1564713, 50.1705475
30: -43.7549667, 14.4203606, -43.7473068, 14.4449816, -58.1999474, 58.1676674
31: -41.6097145, 2.8602581, -41.6292114, 2.8796721, -44.4893875, 44.4894714
32: -38.7558784, 22.5508785, -38.7630310, 22.5258045, -61.2816849, 61.3139114
33: -19.6587200, 60.0551910, -19.7031212, 60.0863609, -73.4040833, 73.3970795
34: -28.2085114, 47.5780983, -28.2480278, 47.6090622, -70.4222031, 70.3796082
35: -18.5745773, 56.0871201, -18.5821495, 56.0824089, -71.5187759, 71.5123062
36: -27.2630005, 48.3201485, -27.3068085, 48.3007812, -73.6086578, 73.6632614
37: -14.7029839, 48.4264374, -14.7613049, 48.4603271, -54.7865410, 54.7850571
38: -33.2593422, 57.7479401, -33.2833939, 57.7121658, -87.4235535, 87.4851913
39: -19.6700745, 65.8449097, -19.6637630, 65.8334427, -77.2900238, 77.2888260
40: -22.8354340, 42.3209267, -22.8585091, 42.3182182, -61.7053566, 61.6811028
41: -26.0468769, 26.3185101, -26.0693474, 26.2835045, -52.3303833, 52.3878555
42: -35.6406021, 19.5815926, -35.6569748, 19.5561371, -55.1967392, 55.2385674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=255, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1141
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1110
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 1140
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 1195
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: A, layer: 1, pos: 1164
type: B, layer: 1, pos: 1164
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1196
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1163
type: A, layer: 1, pos: 1163
type: B, layer: 1, pos: 1160
type: A, layer: 1, pos: 1160
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1142
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1222
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 1224
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 1095
type: B, layer: 1, pos: 1095
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1156
type: B, layer: 1, pos: 1156
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1208
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: A, layer: 1, pos: 1169
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1227
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1235
type: A, layer: 1, pos: 1235
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1109
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1250
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1170
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1263
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1105
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1106
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1090
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1219
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1509
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1186
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1104
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1386
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1120
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1448
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1121
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1202
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 616

## Relational analysis of IS_A1_B1_A2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.1248790, upper bound: 39.1335882
time: 82.37 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.1221318, upper bound: 39.2701360
time: 72.25 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -63.2746773, 8.4676466, -63.4644470, 8.6386805, -71.9133606, 71.9320908
1: -44.4870911, 6.5470743, -44.6155396, 6.7576437, -48.3305397, 48.2266655
2: -35.2788239, 10.6143341, -35.3762856, 10.7274694, -43.4586945, 43.4265747
3: -45.9717636, 7.3832130, -46.0722427, 7.4897394, -50.0687790, 50.0373611
4: -36.1079330, 19.5577316, -36.2336807, 19.7334766, -54.5718803, 54.4874725
5: -49.1330223, 11.0411396, -49.2399445, 11.1461267, -54.9065018, 54.8974495
6: -43.4445953, 19.0553284, -43.6280975, 19.1407528, -62.5853500, 62.6834259
7: -66.7386627, -0.4551849, -66.8780212, -0.2850037, -55.6801147, 55.6040192
8: -42.6613846, 24.8703594, -42.8338089, 25.0801868, -67.7415695, 67.7041702
9: -21.5678329, 16.4211826, -21.6848431, 16.5548477, -38.1226807, 38.1060257
10: -53.1124458, 17.3620644, -53.2553940, 17.6121731, -68.5757751, 68.4545135
11: -69.5466766, -11.6173954, -69.6669312, -11.4711323, -46.8172455, 46.7792511
12: -32.5283318, 30.0095940, -32.6374588, 30.0978851, -59.7876358, 59.8271751
13: -35.7895432, 37.2485886, -36.0229416, 37.3677979, -73.1573410, 73.2715302
14: -105.2255249, -11.1338100, -105.4821930, -10.7992992, -89.1871414, 89.0799789
15: -35.1880798, 21.8325958, -35.2972107, 22.0233269, -56.8643341, 56.7593460
16: -60.9141655, 2.2278099, -61.0555954, 2.4198112, -54.9159241, 54.8712769
17: -122.8154144, -17.7559166, -123.0197449, -17.4337330, -82.3950348, 82.2858505
18: -46.8801804, 24.0869713, -47.0195999, 24.3351707, -71.2153473, 71.1065674
19: -40.1845627, 1.7058368, -40.2624435, 1.7933006, -39.5866127, 39.5800858
20: -31.6419430, 5.3648844, -31.7318478, 5.4356842, -37.0776291, 37.0967331
21: -53.1668549, 0.1589937, -53.2841530, 0.2961884, -49.8081131, 49.7974434
22: -53.8937340, 6.0472174, -54.0182419, 6.2117157, -55.6791801, 55.6326637
23: -32.7855606, 8.2540607, -32.8488922, 8.3440742, -38.7306633, 38.7329063
24: -26.0201359, 18.4481087, -26.0998840, 18.5797501, -44.0347977, 43.9871521
25: -23.3675194, 19.6646767, -23.4812336, 19.7994862, -41.6111526, 41.5920944
26: -44.8489761, 25.1915092, -45.0135269, 25.4048576, -70.2538300, 70.2050323
27: -45.6805344, 10.9301624, -45.7601433, 11.0614376, -56.7419739, 56.6903076
28: -36.1522369, 14.3651123, -36.2158890, 14.4397678, -48.8183670, 48.8329315
29: -65.1995850, -6.1939373, -65.3255310, -6.0309067, -50.1688538, 50.1240311
30: -43.7468719, 14.3927002, -43.8261070, 14.5496521, -58.2965240, 58.2188072
31: -41.5859032, 2.8330169, -41.6759453, 2.9454665, -44.5313683, 44.5089607
32: -38.7106247, 22.4564133, -38.8940544, 22.5494270, -61.2600517, 61.3504677
33: -19.5693436, 59.9216118, -19.9035454, 60.0886803, -73.3205872, 73.5052185
34: -28.1522865, 47.4642563, -28.3508987, 47.6033936, -70.3746414, 70.4587860
35: -18.5073509, 55.9570694, -18.7337303, 56.0796967, -71.4492798, 71.5272980
36: -27.1886635, 48.1579475, -27.4751148, 48.2966423, -73.5351410, 73.6842499
37: -14.6297951, 48.3641357, -14.9174042, 48.4580536, -54.6468048, 54.8636360
38: -33.1615524, 57.5732193, -33.4969940, 57.7327805, -87.3596954, 87.5139847
39: -19.5811863, 65.6723557, -19.8841648, 65.8231049, -77.1893921, 77.3501968
40: -22.7845764, 42.2237511, -22.9933395, 42.3261909, -61.6753006, 61.7888641
41: -26.0015297, 26.2478313, -26.1899986, 26.3141689, -52.3156967, 52.4378281
42: -35.6033173, 19.5293922, -35.7523117, 19.5941849, -55.1975021, 55.2817039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=255, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1111
type: A, layer: 1, pos: 1111
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1141
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1110
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: A, layer: 1, pos: 1125
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1140
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1126
type: A, layer: 1, pos: 1126
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1211
type: A, layer: 1, pos: 1211
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1195
type: A, layer: 1, pos: 1195
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1223
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1401
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1164
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1180
type: A, layer: 1, pos: 1180
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1179
type: A, layer: 1, pos: 1179
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1163
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1160
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1142
type: A, layer: 1, pos: 1142
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1222
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 1224
type: A, layer: 1, pos: 1224
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1107
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1212
type: A, layer: 1, pos: 1212
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1148
type: B, layer: 1, pos: 1148
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1095
type: A, layer: 1, pos: 1095
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1156
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1092
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1208
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1208
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1091
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1169
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1247
type: B, layer: 1, pos: 1247
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1227
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1123
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1231
type: A, layer: 1, pos: 1231
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1109
type: A, layer: 1, pos: 1109
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1122
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1516
type: A, layer: 1, pos: 1516
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1250
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1138
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1279
type: B, layer: 1, pos: 1279
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1170
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1230
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1263
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1139
type: B, layer: 1, pos: 1139
type: A, layer: 1, pos: 1106
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1218
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1127
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1090
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1132
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1153
type: B, layer: 1, pos: 1153
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1219
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1201
type: B, layer: 1, pos: 1201
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1187
type: A, layer: 1, pos: 1187
type: B, layer: 1, pos: 1509
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1186
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1104
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1136
type: B, layer: 1, pos: 1136
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1120
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1448
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: B, layer: 1, pos: 1121
type: A, layer: 1, pos: 1185
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1168
type: B, layer: 1, pos: 1168
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1215
type: A, layer: 1, pos: 1215
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1203
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1200
type: B, layer: 1, pos: 1200
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1137
type: B, layer: 1, pos: 1137
type: A, layer: 1, pos: 1184
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1152
type: B, layer: 1, pos: 1152
type: A, layer: 1, pos: 1526
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1202
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 616

## Relational analysis of IS_A1_B2_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2998901, upper bound: 39.2432975
time: 53.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2971679, upper bound: 39.2432992
time: 41.24 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 96.99 seconds
IS_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 96.99
Output dim: 37, lower bound: -39.2849094, upper bound: 39.2432975
IS_A1_B1_A1_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 96.99
Output dim: 37, lower bound: -39.1103776, upper bound: 39.2432992
IS_A1_B1_A2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 96.99
Output dim: 37, lower bound: -39.1248790, upper bound: 39.1335882
IS_A1_B1_A2_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 96.99
Output dim: 37, lower bound: -39.1221318, upper bound: 39.2701360
IS_A1_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 96.99
Output dim: 37, lower bound: -39.2998901, upper bound: 39.2432975
IS_A1_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 96.99
Output dim: 37, lower bound: -39.2971679, upper bound: 39.2432992
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 96.99
Output dim: 37, lower bound: -39.3011099, upper bound: 39.2740781
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 96.99
Output dim: 37, lower bound: -39.2861286, upper bound: 39.2742629
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 96.99
Output dim: 37, lower bound: -39.3011099, upper bound: 39.2742629

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 100.29 + 3575.53 = 3675.82 seconds
