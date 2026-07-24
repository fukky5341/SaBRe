## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 4)
Time budget: 7200 seconds
Split limit: 100
Threshold: 142.787795174


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=698, inp2_unstable=698, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=874, inp2_unstable=874, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-194.3560638, 123.7643051, -194.3560638, 123.7643051, -318.1203613, 318.1203613)
1: (-112.6573334, 98.2656555, -112.6573334, 98.2656555, -210.9229889, 210.9229889)
2: (-99.1251068, 90.8807373, -99.1251068, 90.8807373, -190.0058441, 190.0058441)
3: (-115.3852997, 118.0533752, -115.3852997, 118.0533752, -233.4386597, 233.4386597)
4: (-122.4373779, 112.1905746, -122.4373779, 112.1905746, -234.6279297, 234.6279602)
5: (-111.2773132, 118.3461990, -111.2773132, 118.3461990, -229.6235046, 229.6234894)
6: (-137.7839966, 120.7834320, -137.7839966, 120.7834320, -258.5674133, 258.5674438)
7: (-135.4126282, 119.6500015, -135.4126282, 119.6500015, -255.0626221, 255.0626221)
8: (-127.4194870, 132.8383179, -127.4194870, 132.8383179, -260.2578125, 260.2578125)
9: (-119.2357254, 110.2499542, -119.2357254, 110.2499542, -229.4856873, 229.4856873)
10: (-174.5004272, 163.4277802, -174.5004272, 163.4277802, -337.9282227, 337.9281921)
11: (-166.8539734, 129.6512146, -166.8539734, 129.6512146, -296.5051880, 296.5051880)
12: (-156.0662231, 146.7076263, -156.0662231, 146.7076263, -302.7738647, 302.7738342)
13: (-161.5158081, 168.4806824, -161.5158081, 168.4806824, -329.9964905, 329.9964905)
14: (-238.9123383, 147.0448761, -238.9123383, 147.0448761, -385.9572144, 385.9572144)
15: (-139.2399445, 112.0155334, -139.2399445, 112.0155334, -251.2554779, 251.2554779)
16: (-182.3288879, 134.4130859, -182.3288879, 134.4130859, -316.7419739, 316.7419739)
17: (-249.6140747, 196.0419617, -249.6140747, 196.0419617, -445.6560059, 445.6560059)
18: (-152.5888062, 126.4845505, -152.5888062, 126.4845505, -279.0733643, 279.0733643)
19: (-117.1988754, 74.9498978, -117.1988754, 74.9498978, -192.1487732, 192.1487732)
20: (-110.0395050, 92.9520416, -110.0395050, 92.9520416, -202.9915466, 202.9915466)
21: (-149.0595856, 100.4649124, -149.0595856, 100.4649124, -249.5244598, 249.5244598)
22: (-158.1970825, 108.0006943, -158.1970825, 108.0006943, -266.1977539, 266.1977844)
23: (-120.2197723, 95.0907288, -120.2197723, 95.0907288, -215.3104858, 215.3104706)
24: (-147.7882538, 108.8617859, -147.7882538, 108.8617859, -256.6500244, 256.6500244)
25: (-124.3082123, 106.1959991, -124.3082123, 106.1959991, -230.5042114, 230.5042114)
26: (-171.0725708, 148.6687469, -171.0725708, 148.6687469, -319.7413330, 319.7413330)
27: (-155.3504333, 110.9619522, -155.3504333, 110.9619522, -266.3123779, 266.3123779)
28: (-116.1600723, 100.0528336, -116.1600723, 100.0528336, -216.2129059, 216.2129059)
29: (-174.3669586, 115.8534164, -174.3669586, 115.8534164, -290.2203674, 290.2203674)
30: (-146.8119354, 125.6781540, -146.8119354, 125.6781540, -272.4900513, 272.4900513)
31: (-149.9770966, 101.8592148, -149.9770966, 101.8592148, -251.8362885, 251.8363037)
32: (-146.5981750, 117.1200714, -146.5981750, 117.1200714, -263.7182617, 263.7182617)
33: (-192.7234802, 159.2073059, -192.7234802, 159.2073059, -351.9307861, 351.9307861)
34: (-161.5019989, 124.2549896, -161.5019989, 124.2549896, -285.7569885, 285.7569885)
35: (-163.2583618, 126.8004913, -163.2583618, 126.8004913, -290.0588379, 290.0588379)
36: (-154.2919922, 125.1236725, -154.2919922, 125.1236725, -279.4156189, 279.4156494)
37: (-223.7451019, 141.9507141, -223.7451019, 141.9507141, -365.6958008, 365.6958008)
38: (-190.3866577, 154.8811646, -190.3866577, 154.8811646, -345.2678223, 345.2678223)
39: (-226.4278870, 154.2048950, -226.4278870, 154.2048950, -380.6327820, 380.6327820)
40: (-192.2375641, 133.5492554, -192.2375641, 133.5492554, -325.7868042, 325.7868042)
41: (-144.3735046, 111.2714844, -144.3735046, 111.2714844, -255.6449738, 255.6449738)
42: (-111.1141891, 108.9842834, -111.1141891, 108.9842834, -220.0984344, 220.0984497)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.94 + 215.72 = 218.66 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -142.9307259, upper bound: 142.9307259

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1238
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1241
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1080
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1166
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1182
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1064
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1081
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1046
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1213
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1077
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1150
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1133
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1073
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1226
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1162
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1225
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1205
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1045
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1083
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1053
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1175
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1060
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 1084
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1058
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1062
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1044
type: A, layer: 1, pos: 1098
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1061
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 260
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 1042
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 307
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 324
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 264
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 308
type: A, layer: 1, pos: 259
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 325
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 291
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1097
type: A, layer: 1, pos: 1078
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 256
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1178
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 257
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 321
type: A, layer: 1, pos: 277
type: A, layer: 1, pos: 336
type: A, layer: 1, pos: 294
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 258
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 305
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 337
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 353
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 293
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 306
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 1792
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 640

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -142.8955464, upper bound: 142.9273231
time: 348.32 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -142.8955464, upper bound: 142.9273231
time: 515.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 863.98 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 863.98
Output dim: 3, lower bound: -142.8955464, upper bound: 142.9273231
IS_A2, status: Status.UNKNOWN, split count: 1, time: 863.98
Output dim: 3, lower bound: -142.8955464, upper bound: 142.9273231

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -193.7549591, 123.4849701, -194.0720520, 123.6533203, -317.4082642, 317.5570068
1: -112.2062836, 98.0295868, -112.4242401, 98.1737671, -210.3800354, 210.4538269
2: -98.4843063, 90.5573730, -98.7688675, 90.8119202, -189.2962189, 189.3262329
3: -114.5677872, 117.5146637, -114.9225159, 117.9415512, -232.5092926, 232.4371490
4: -121.6564331, 111.7990570, -122.0077896, 112.1169510, -233.7733765, 233.8068542
5: -110.6400604, 117.8679428, -110.9254837, 118.2417450, -228.8817902, 228.7934113
6: -137.5158997, 120.3617401, -137.6589050, 120.5777969, -258.0936890, 258.0206299
7: -134.8545532, 119.3860931, -135.1254883, 119.5683899, -254.4229431, 254.5115814
8: -126.7255859, 132.4398193, -127.0365143, 132.7318726, -259.4574585, 259.4763184
9: -118.8068085, 109.9080811, -119.0373840, 110.1120682, -228.9188843, 228.9454651
10: -173.7842712, 162.0623169, -174.3294678, 162.6706238, -336.4548340, 336.3917847
11: -166.1309052, 128.3669128, -166.7203369, 128.8924713, -295.0233765, 295.0872192
12: -155.3451996, 145.5450287, -155.9434967, 146.0529480, -301.3981323, 301.4885254
13: -160.7512817, 167.9092712, -161.1126099, 168.2918701, -329.0431213, 329.0218811
14: -238.0004883, 145.9578552, -238.6828308, 146.4176941, -384.4181519, 384.6406860
15: -138.4383240, 111.6974487, -138.7922668, 111.8832550, -250.3215790, 250.4896851
16: -181.7315674, 133.5110779, -182.0875549, 133.9145203, -315.6460876, 315.5986328
17: -248.7440643, 194.5653534, -249.4511871, 195.1947327, -443.9387817, 444.0165405
18: -152.0092010, 125.4978790, -152.3652191, 125.9005890, -277.9097900, 277.8630981
19: -116.7126541, 74.3825836, -117.0648804, 74.6199951, -191.3326263, 191.4474640
20: -109.6524963, 92.4611664, -109.9106979, 92.6810608, -202.3335266, 202.3718262
21: -148.4987640, 99.6449738, -148.9260254, 99.9935608, -248.4923248, 248.5709991
22: -157.7925415, 107.4088287, -158.0611877, 107.6801300, -265.4726562, 265.4700012
23: -119.7585754, 94.4587250, -120.1069565, 94.7329254, -214.4915009, 214.5656738
24: -147.2999573, 108.3781128, -147.6360931, 108.5942078, -255.8941193, 256.0141907
25: -123.8804703, 105.6260529, -124.1865234, 105.8795776, -229.7600403, 229.8125763
26: -170.4028320, 147.5608826, -170.9080811, 148.0549316, -318.4577637, 318.4689636
27: -154.9410706, 110.6074066, -155.1665802, 110.7758331, -265.7168579, 265.7739868
28: -115.8267670, 99.7016830, -116.0326462, 99.8703308, -215.6970978, 215.7343292
29: -173.9377441, 115.0544586, -174.2609253, 115.3997192, -289.3374634, 289.3153687
30: -146.3535004, 124.7689056, -146.6912384, 125.1696930, -271.5231628, 271.4601440
31: -149.3532104, 101.1433334, -149.7796478, 101.4455795, -250.7987671, 250.9229736
32: -146.2322388, 116.7891159, -146.4330444, 116.9579773, -263.1902161, 263.2221375
33: -191.8767242, 158.7216187, -192.2612762, 159.1068420, -350.9835815, 350.9828491
34: -160.9298859, 123.9075241, -161.1934662, 124.1378021, -285.0676575, 285.1009827
35: -162.4543152, 126.3980789, -162.8134918, 126.7237244, -289.1780090, 289.2115784
36: -153.6648712, 124.8371429, -153.9573669, 125.0384293, -278.7032776, 278.7944946
37: -223.1360931, 141.5203857, -223.4708557, 141.7457275, -364.8817749, 364.9912415
38: -189.6068726, 154.5263062, -189.9816895, 154.7505798, -344.3574524, 344.5079651
39: -225.5913849, 153.8287659, -225.9892273, 154.1257477, -379.7171326, 379.8179626
40: -191.6238251, 133.3120422, -191.9257812, 133.4826965, -325.1065063, 325.2378235
41: -143.9954071, 110.9291992, -144.1990356, 111.1210327, -255.1164246, 255.1282349
42: -110.8106079, 108.2577133, -111.0244675, 108.5943451, -219.4049377, 219.2821808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=697, inp2_unstable=698, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=874, inp2_unstable=874, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1238
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1241
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1242
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1155
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1080
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1166
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1182
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1064
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1048
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1081
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1046
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1213
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1077
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1150
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1183
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1133
type: B, layer: 1, pos: 1073
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1226
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1243
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1162
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1225
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1205
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1059
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1045
type: B, layer: 1, pos: 1083
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1053
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1175
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1060
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 1084
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1058
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1062
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1098
type: B, layer: 1, pos: 1044
type: B, layer: 1, pos: 263
type: B, layer: 1, pos: 1057
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1061
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 260
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1067
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1074
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1042
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 324
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 309
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 264
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 308
type: B, layer: 1, pos: 259
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 325
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 291
type: B, layer: 1, pos: 1043
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1097
type: B, layer: 1, pos: 1078
type: B, layer: 1, pos: 1068
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 256
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1178
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 257
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 321
type: B, layer: 1, pos: 277
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 336
type: B, layer: 1, pos: 294
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 258
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 305
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 337
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 353
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 293
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 306
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 1792
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1259

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -142.8932230, upper bound: 142.8847697
time: 750.76 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -142.8933735, upper bound: 142.9251085
time: 289.39 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -194.3078918, 123.7332611, -194.3273773, 123.7466888, -318.0545654, 318.0606079
1: -112.6309509, 98.2404327, -112.6420135, 98.2507324, -210.8816528, 210.8824463
2: -99.0969391, 90.8699188, -99.1087723, 90.8744888, -189.9714355, 189.9786987
3: -115.3410339, 118.0351181, -115.3599472, 118.0427094, -233.3837433, 233.3950653
4: -122.3941040, 112.1770630, -122.4123383, 112.1827316, -234.5768280, 234.5893860
5: -111.2417908, 118.3307190, -111.2568970, 118.3371658, -229.5789490, 229.5876160
6: -137.7627258, 120.6949234, -137.7717285, 120.7332535, -258.4959412, 258.4666443
7: -135.3790588, 119.6356812, -135.3933411, 119.6417465, -255.0208130, 255.0290222
8: -127.3813705, 132.8206482, -127.3974762, 132.8281860, -260.2095337, 260.2181091
9: -119.1719818, 110.2248077, -119.1992264, 110.2353516, -229.4073029, 229.4240417
10: -174.4777222, 163.3535614, -174.4872437, 163.3849182, -337.8626099, 337.8407593
11: -166.8300781, 129.5824585, -166.8402252, 129.6116180, -296.4417114, 296.4226685
12: -156.0473022, 146.6483459, -156.0552673, 146.6734619, -302.7207642, 302.7036133
13: -161.4288025, 168.4491882, -161.4613037, 168.4625854, -329.8913879, 329.9104919
14: -238.8775940, 146.9903412, -238.8922119, 147.0136261, -385.8911743, 385.8825684
15: -139.2107544, 111.9941559, -139.2225952, 112.0031891, -251.2139435, 251.2167511
16: -182.2944336, 134.3719482, -182.3087158, 134.3868103, -316.6812439, 316.6806641
17: -249.5905762, 195.9656677, -249.6002350, 195.9980469, -445.5886230, 445.5659180
18: -152.5502014, 126.4285126, -152.5665283, 126.4518738, -279.0020752, 278.9950562
19: -117.1812973, 74.9178162, -117.1886826, 74.9315338, -192.1128235, 192.1065063
20: -110.0221634, 92.9320831, -110.0295334, 92.9403687, -202.9625244, 202.9616089
21: -149.0392609, 100.4221573, -149.0478516, 100.4403381, -249.4795532, 249.4700012
22: -158.1729431, 107.9643173, -158.1831970, 107.9796677, -266.1526184, 266.1474915
23: -120.2049866, 95.0537720, -120.2112198, 95.0694885, -215.2744598, 215.2649841
24: -147.7589722, 108.7959595, -147.7713318, 108.8244171, -256.5833740, 256.5672913
25: -124.2901840, 106.1631775, -124.2977448, 106.1771088, -230.4672852, 230.4608917
26: -171.0451050, 148.6124725, -171.0567932, 148.6363220, -319.6814270, 319.6692505
27: -155.3201752, 110.8967896, -155.3329468, 110.9249039, -266.2450867, 266.2297363
28: -116.1407166, 100.0255280, -116.1489182, 100.0372009, -216.1779022, 216.1744385
29: -174.3449860, 115.8067093, -174.3542480, 115.8264771, -290.1714478, 290.1609497
30: -146.7911530, 125.6270905, -146.7999268, 125.6488190, -272.4399719, 272.4270020
31: -149.9496460, 101.8212433, -149.9612579, 101.8374939, -251.7871399, 251.7824860
32: -146.5530548, 117.0799866, -146.5723724, 117.0965118, -263.6495361, 263.6523438
33: -192.6744690, 159.1912537, -192.6951141, 159.1980591, -351.8724976, 351.8863525
34: -161.4667969, 124.2390137, -161.4817505, 124.2456818, -285.7124329, 285.7207642
35: -163.2134094, 126.7888718, -163.2324524, 126.7938004, -290.0072021, 290.0213013
36: -154.2553101, 125.1101837, -154.2709198, 125.1158447, -279.3711548, 279.3811035
37: -223.6952362, 141.8906860, -223.7158813, 141.9166107, -365.6118469, 365.6065674
38: -190.3352661, 154.8501587, -190.3570251, 154.8632965, -345.1985474, 345.2071838
39: -226.3760681, 154.1922302, -226.3977051, 154.1976624, -380.5737305, 380.5899353
40: -192.1906281, 133.5386047, -192.2102509, 133.5431213, -325.7337646, 325.7488403
41: -144.3476410, 111.2127991, -144.3585052, 111.2379074, -255.5855408, 255.5712891
42: -111.1004257, 108.9377899, -111.1062469, 108.9565582, -220.0569763, 220.0440369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=697, inp2_unstable=698, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=874, inp2_unstable=874, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1238
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1241
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1242
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1155
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1080
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1166
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1182
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1064
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1048
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1081
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1046
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1213
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1077
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1150
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1183
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1133
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1073
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1226
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1243
type: B, layer: 1, pos: 1162
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1225
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1205
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1059
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1045
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 1083
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1053
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1175
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1060
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 1084
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1058
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1062
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1044
type: B, layer: 1, pos: 1098
type: B, layer: 1, pos: 263
type: B, layer: 1, pos: 1057
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1061
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 260
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1067
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1074
type: B, layer: 1, pos: 1042
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 324
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 309
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 264
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 308
type: B, layer: 1, pos: 259
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 325
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 291
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1043
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1097
type: B, layer: 1, pos: 1078
type: B, layer: 1, pos: 1068
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 256
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1178
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 257
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 321
type: B, layer: 1, pos: 277
type: B, layer: 1, pos: 336
type: B, layer: 1, pos: 294
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 258
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 305
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 337
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 353
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 293
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 306
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 1792
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 640

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -142.8932230, upper bound: 142.8847697
time: 1104.31 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -142.9251084, upper bound: 142.9251085
time: 1834.87 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2941.75 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2941.75
Output dim: 3, lower bound: -142.8932230, upper bound: 142.8847697
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2941.75
Output dim: 3, lower bound: -142.8933735, upper bound: 142.9251085
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2941.75
Output dim: 3, lower bound: -142.8932230, upper bound: 142.8847697
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2941.75
Output dim: 3, lower bound: -142.9251084, upper bound: 142.9251085

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -193.4098663, 123.4053802, -193.4101410, 123.3989105, -316.8087769, 316.8155212
1: -111.9604340, 97.9668655, -111.9682465, 98.0056152, -209.9660492, 209.9350891
2: -98.0948029, 90.5077820, -98.0668945, 90.5420837, -188.6368866, 188.5746765
3: -114.1237564, 117.4289856, -114.1284485, 117.5828171, -231.7065735, 231.5574341
4: -121.2269745, 111.7332153, -121.2275009, 111.7772064, -233.0041656, 232.9607239
5: -110.2398376, 117.7890701, -110.2015762, 117.8839569, -228.1237946, 227.9906464
6: -137.3885956, 120.1320343, -137.3872375, 120.1283417, -257.5169373, 257.5192566
7: -134.5062561, 119.3188629, -134.4710236, 119.3178711, -253.8241272, 253.7898712
8: -126.2900238, 132.3600159, -126.2500763, 132.3914795, -258.6814880, 258.6100769
9: -118.7078934, 109.6407318, -118.7859421, 109.5881958, -228.2960815, 228.4266357
10: -173.6285858, 161.3134003, -173.7235107, 161.3274231, -334.9559937, 335.0369263
11: -166.0110474, 127.7573013, -166.2580566, 127.8206482, -293.8316650, 294.0153503
12: -155.2554321, 144.7580719, -155.3286438, 144.6548462, -299.9102783, 300.0867004
13: -160.6159973, 167.7176361, -160.8209534, 167.8980408, -328.5140381, 328.5385742
14: -237.8147430, 145.4831696, -238.0638885, 145.5694733, -383.3842163, 383.5470581
15: -138.1027222, 111.5623398, -138.1438141, 111.6053085, -249.7080383, 249.7061157
16: -181.5366364, 133.0594788, -181.6522369, 133.0798035, -314.6164246, 314.7117310
17: -248.6189270, 193.8825989, -248.9095764, 193.9638062, -442.5827026, 442.7921753
18: -151.8176880, 125.1450195, -151.9206848, 125.2531281, -277.0708008, 277.0657043
19: -116.5956573, 74.1782379, -116.7519073, 74.2491531, -190.8448029, 190.9301147
20: -109.5301361, 92.2830734, -109.6079254, 92.3593445, -201.8894653, 201.8909912
21: -148.3820801, 99.3305664, -148.5520477, 99.4279327, -247.8100128, 247.8826141
22: -157.6448669, 107.1555176, -157.7167053, 107.1955719, -264.8404541, 264.8722229
23: -119.6599884, 94.2793045, -119.8439178, 94.4024048, -214.0623932, 214.1232300
24: -147.1420593, 108.3298340, -147.3134918, 108.4855652, -255.6276245, 255.6433105
25: -123.7894974, 105.4373627, -123.9532776, 105.5142593, -229.3037262, 229.3906403
26: -170.2640991, 147.0176392, -170.4228516, 147.0700836, -317.3341064, 317.4404907
27: -154.6963196, 110.5530243, -154.6885376, 110.6315613, -265.3278809, 265.2415771
28: -115.7007828, 99.6264648, -115.7556229, 99.7048950, -215.4056702, 215.3820801
29: -173.8335571, 114.7058105, -173.9875793, 114.7701111, -288.6036377, 288.6933899
30: -146.2503510, 124.4943390, -146.4466248, 124.6498413, -270.9001770, 270.9409790
31: -149.1770630, 100.9244156, -149.3663330, 101.0388794, -250.2159271, 250.2907257
32: -146.1315613, 116.5549850, -146.1485138, 116.5348816, -262.6664429, 262.7034912
33: -191.5147400, 158.6183319, -191.5892181, 158.7653046, -350.2800293, 350.2075500
34: -160.6600342, 123.8054047, -160.6825256, 123.8715973, -284.5316162, 284.4879150
35: -162.0951538, 126.3255539, -162.1575012, 126.4449310, -288.5401001, 288.4830627
36: -153.4520569, 124.7519379, -153.5346527, 124.8467941, -278.2988281, 278.2865906
37: -222.9294739, 141.3445129, -223.0219269, 141.3900757, -364.3195496, 364.3664246
38: -189.2800598, 154.4250641, -189.3438721, 154.5139923, -343.7940674, 343.7689209
39: -225.3232269, 153.7533569, -225.4561768, 153.8957825, -379.2189941, 379.2095337
40: -191.3724365, 133.2558899, -191.4244232, 133.3330688, -324.7054749, 324.6802979
41: -143.8566895, 110.7810516, -143.9247437, 110.8209152, -254.6776123, 254.7057648
42: -110.7266998, 107.8401031, -110.6912384, 107.8346405, -218.5613251, 218.5313110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=697, inp2_unstable=697, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=874, inp2_unstable=874, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1238
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1241
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1080
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1166
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1182
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1064
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1081
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1046
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1213
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1077
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1150
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1133
type: A, layer: 1, pos: 1073
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1226
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1162
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1225
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1205
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1083
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1045
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1053
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1175
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1060
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 1084
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1058
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1062
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1098
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 1044
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1061
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 260
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1042
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 307
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 324
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 264
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 308
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 259
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 325
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 291
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1097
type: A, layer: 1, pos: 1078
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 256
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1178
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 257
type: A, layer: 1, pos: 321
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 277
type: A, layer: 1, pos: 336
type: A, layer: 1, pos: 294
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 258
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 305
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 337
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 353
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 293
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 306
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 1792
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 640

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -142.8661105, upper bound: 142.7985149
time: 285.43 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -142.8661105, upper bound: 142.8807087
time: 468.04 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -193.7380829, 123.4819565, -194.0408325, 123.6475830, -317.3856812, 317.5227966
1: -112.1947327, 98.0271149, -112.4031219, 98.1691132, -210.3638306, 210.4302216
2: -98.4702301, 90.5543671, -98.7426529, 90.8063889, -189.2766113, 189.2970276
3: -114.5511780, 117.5099564, -114.8909302, 117.9328842, -232.4840546, 232.4008789
4: -121.6410065, 111.7947235, -121.9792252, 112.1088486, -233.7498474, 233.7739258
5: -110.6250839, 117.8640442, -110.8969803, 118.2345276, -228.8595886, 228.7610168
6: -137.5084381, 120.3324051, -137.6448669, 120.5211487, -258.0296021, 257.9772644
7: -134.8406830, 119.3827820, -135.0991974, 119.5622864, -254.4029541, 254.4819641
8: -126.7095337, 132.4357910, -127.0065613, 132.7244568, -259.4339905, 259.4423523
9: -118.8011169, 109.8986588, -119.0268402, 110.0942383, -228.8953247, 228.9254761
10: -173.7766418, 162.0369263, -174.3153229, 162.6232910, -336.3999329, 336.3522339
11: -166.1236572, 128.3474121, -166.7069855, 128.8557739, -294.9794312, 295.0543823
12: -155.3406830, 145.5199585, -155.9350586, 146.0050354, -301.3457031, 301.4550171
13: -160.7301178, 167.8997955, -161.0719910, 168.2740784, -329.0041504, 328.9718018
14: -237.9909973, 145.9429321, -238.6653748, 146.3887939, -384.3797913, 384.6083069
15: -138.4042511, 111.6905975, -138.7374268, 111.8703537, -250.2745972, 250.4280243
16: -181.7201233, 133.4932098, -182.0661926, 133.8808441, -315.6009521, 315.5593872
17: -248.7380676, 194.5437012, -249.4400635, 195.1534424, -443.8914795, 443.9837646
18: -151.9987793, 125.4851074, -152.3451080, 125.8761597, -277.8749084, 277.8302002
19: -116.7069855, 74.3744965, -117.0543747, 74.6052017, -191.3121796, 191.4288635
20: -109.6477509, 92.4538651, -109.9017639, 92.6680145, -202.3157654, 202.3556213
21: -148.4924927, 99.6343842, -148.9143677, 99.9737091, -248.4662018, 248.5487366
22: -157.7780609, 107.3960037, -158.0338135, 107.6556244, -265.4336548, 265.4298096
23: -119.7537537, 94.4463806, -120.0979767, 94.7097473, -214.4635010, 214.5443573
24: -147.2879333, 108.3743591, -147.6104126, 108.5870819, -255.8749695, 255.9847717
25: -123.8750305, 105.6185760, -124.1765747, 105.8656006, -229.7406311, 229.7951050
26: -170.3961182, 147.5419617, -170.8956757, 148.0187073, -318.4148254, 318.4376221
27: -154.9296265, 110.6035080, -155.1453552, 110.7686005, -265.6982422, 265.7488708
28: -115.8191910, 99.6949158, -116.0181961, 99.8578796, -215.6770630, 215.7131042
29: -173.9308319, 115.0410919, -174.2484741, 115.3742752, -289.3050842, 289.2895508
30: -146.3473511, 124.7437668, -146.6800842, 125.1212540, -271.4685974, 271.4238281
31: -149.3453979, 101.1343079, -149.7650757, 101.4289780, -250.7743835, 250.8993835
32: -146.2262726, 116.7779846, -146.4220276, 116.9398041, -263.1660767, 263.2000122
33: -191.8644104, 158.7162018, -192.2376099, 159.0967712, -350.9611816, 350.9537964
34: -160.9199829, 123.9023285, -161.1745148, 124.1284027, -285.0483704, 285.0768127
35: -162.4417419, 126.3942871, -162.7897339, 126.7166824, -289.1584167, 289.1840210
36: -153.6552582, 124.8325195, -153.9393005, 125.0297089, -278.6849365, 278.7718201
37: -223.1242065, 141.5061035, -223.4488525, 141.7186279, -364.8428040, 364.9549561
38: -189.5936584, 154.5199738, -189.9569702, 154.7386627, -344.3323059, 344.4769287
39: -225.5766907, 153.8247986, -225.9614716, 154.1183777, -379.6950378, 379.7862549
40: -191.6130676, 133.3074646, -191.9054565, 133.4746094, -325.0876465, 325.2129211
41: -143.9884949, 110.9097290, -144.1860962, 111.0836334, -255.0720978, 255.0958252
42: -110.8055420, 108.2385559, -111.0149002, 108.5613937, -219.3669128, 219.2534485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=697, inp2_unstable=697, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=874, inp2_unstable=874, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1238
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1241
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1080
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1166
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1182
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1064
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1081
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1046
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1213
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1077
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1150
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1133
type: A, layer: 1, pos: 1073
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1226
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1162
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1225
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1205
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1045
type: A, layer: 1, pos: 1083
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1053
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1175
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1060
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 1084
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1058
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1062
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1098
type: A, layer: 1, pos: 1044
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1061
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 260
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1042
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 307
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 324
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 264
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 308
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 259
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 325
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 291
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1097
type: A, layer: 1, pos: 1078
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 256
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1178
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 257
type: A, layer: 1, pos: 321
type: A, layer: 1, pos: 277
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 336
type: A, layer: 1, pos: 294
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 258
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 305
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 337
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 353
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 293
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 306
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 1792
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 640

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -142.8662077, upper bound: 142.8387701
time: 738.49 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -142.8662077, upper bound: 142.8387701
time: 3481.96 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4222.88 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4222.88
Output dim: 3, lower bound: -142.8661105, upper bound: 142.7985149
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4222.88
Output dim: 3, lower bound: -142.8661105, upper bound: 142.8807087
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4222.88
Output dim: 3, lower bound: -142.8662077, upper bound: 142.8387701
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4222.88
Output dim: 3, lower bound: -142.8662077, upper bound: 142.8387701
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4222.88
Output dim: 3, lower bound: -142.8932230, upper bound: 142.8847697
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4222.88
Output dim: 3, lower bound: -142.9251084, upper bound: 142.9251085

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 218.66 + 9827.12 = 10045.78 seconds
