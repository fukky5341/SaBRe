## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 113.951405901
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094)
1: (-49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728)
2: (-48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477)
3: (-52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180)
4: (-65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904)
5: (-55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736)
6: (-86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349)
7: (-67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533)
8: (-78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369)
9: (-61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378)
10: (-91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857)
11: (-83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308)
12: (-60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689)
13: (-67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264)
14: (-116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854)
15: (-66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329)
16: (-97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562)
17: (-109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844)
18: (-91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036)
19: (-67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806)
20: (-65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906)
21: (-84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383)
22: (-74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077)
23: (-66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527)
24: (-85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511)
25: (-63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815)
26: (-90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034)
27: (-99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819)
28: (-66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736)
29: (-78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234)
30: (-83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396)
31: (-89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892)
32: (-76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247)
33: (-113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937)
34: (-90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572)
35: (-88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323)
36: (-87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383)
37: (-136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212)
38: (-109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264)
39: (-121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669)
40: (-112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069)
41: (-86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731)
42: (-58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549)

## BASE Result
execution time: IAR + LP analysis = 2.98 + 155.87 = 158.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 13, lower bound: -123.2820485, upper bound: 123.2820485


# Binary Search by BASE starts (time budget: 17841.15 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=172.8270263671875
rel_dist={13: [-114.05422291258427, 114.05422290962252]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=172.8270263671875
rel_dist={13: [-107.09440620076711, 107.0944062017914]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=172.8270263671875
rel_dist={13: [-109.64944808549234, 109.64944808379073]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=172.8270263671875
rel_dist={13: [-111.95995320337546, 111.95995319383253]}

## Binary Search Result
Binary search time: 507.68 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_dual_Z) starts
Time budget: 17333.47 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1725

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1070378, upper bound: 119.2543410
time: 98.65 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2543410, upper bound: 119.1070378
time: 121.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 220.53 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 220.53
Output dim: 13, lower bound: -119.1070378, upper bound: 119.2543410
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 220.53
Output dim: 13, lower bound: -119.2543410, upper bound: 119.1070378

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1597

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0884695, upper bound: 119.1974547
time: 193.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0505295, upper bound: 119.2355430
time: 100.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1597

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2355430, upper bound: 119.0505295
time: 147.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1974547, upper bound: 119.0884695
time: 112.33 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 262.77 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 262.77
Output dim: 13, lower bound: -119.0884695, upper bound: 119.1974547
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 262.77
Output dim: 13, lower bound: -119.0505295, upper bound: 119.2355430
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 262.77
Output dim: 13, lower bound: -119.2355430, upper bound: 119.0505295
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 262.77
Output dim: 13, lower bound: -119.1974547, upper bound: 119.0884695

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1630

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0842917, upper bound: 119.1029281
time: 117.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -118.9944116, upper bound: 119.1934390
time: 528.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1630

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0464894, upper bound: 119.1412280
time: 89.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -118.9561442, upper bound: 119.2314085
time: 139.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1630

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2314085, upper bound: 118.9561442
time: 101.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1412280, upper bound: 119.0464894
time: 101.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1630

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1934390, upper bound: 118.9944116
time: 99.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1029281, upper bound: 119.0842917
time: 117.24 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 221.56 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 221.56
Output dim: 13, lower bound: -119.0842917, upper bound: 119.1029281
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 221.56
Output dim: 13, lower bound: -118.9944116, upper bound: 119.1934390
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 221.56
Output dim: 13, lower bound: -119.0464894, upper bound: 119.1412280
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 221.56
Output dim: 13, lower bound: -118.9561442, upper bound: 119.2314085
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 221.56
Output dim: 13, lower bound: -119.2314085, upper bound: 118.9561442
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 221.56
Output dim: 13, lower bound: -119.1412280, upper bound: 119.0464894
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 221.56
Output dim: 13, lower bound: -119.1934390, upper bound: 118.9944116
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 221.56
Output dim: 13, lower bound: -119.1029281, upper bound: 119.0842917

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -118.9698314, upper bound: 119.0980904
time: 121.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0795068, upper bound: 118.9886245
time: 742.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -118.8797355, upper bound: 119.1887221
time: 112.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -118.9894504, upper bound: 119.0792893
time: 141.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -118.9319077, upper bound: 119.1364812
time: 124.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.0416094, upper bound: 119.0270725
time: 280.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -118.8412697, upper bound: 119.2267197
time: 109.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -118.9509608, upper bound: 119.1172736
time: 149.50 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 264.91 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 264.91
Output dim: 13, lower bound: -118.9698314, upper bound: 119.0980904
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 264.91
Output dim: 13, lower bound: -119.0795068, upper bound: 118.9886245
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 264.91
Output dim: 13, lower bound: -118.8797355, upper bound: 119.1887221
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 264.91
Output dim: 13, lower bound: -118.9894504, upper bound: 119.0792893
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 264.91
Output dim: 13, lower bound: -118.9319077, upper bound: 119.1364812
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 264.91
Output dim: 13, lower bound: -119.0416094, upper bound: 119.0270725
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 264.91
Output dim: 13, lower bound: -118.8412697, upper bound: 119.2267197
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 264.91
Output dim: 13, lower bound: -118.9509608, upper bound: 119.1172736
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 264.91
Output dim: 13, lower bound: -119.2314085, upper bound: 118.9561442
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 264.91
Output dim: 13, lower bound: -119.1412280, upper bound: 119.0464894
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 264.91
Output dim: 13, lower bound: -119.1934390, upper bound: 118.9944116
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 264.91
Output dim: 13, lower bound: -119.1029281, upper bound: 119.0842917
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=172.8270263671875
rel_dist={13: [-119.26634416186121, 119.2663441598697]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1725

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8132670, upper bound: 115.9344275
time: 112.27 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.9344275, upper bound: 115.8132670
time: 130.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 243.18 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 243.18
Output dim: 13, lower bound: -115.8132670, upper bound: 115.9344275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 243.18
Output dim: 13, lower bound: -115.9344275, upper bound: 115.8132670

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1597

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8071756, upper bound: 115.8852544
time: 137.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7641302, upper bound: 115.9283281
time: 96.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1597

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.9283281, upper bound: 115.7641302
time: 109.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8852544, upper bound: 115.8071756
time: 99.59 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 211.75 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 211.75
Output dim: 13, lower bound: -115.8071756, upper bound: 115.8852544
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 211.75
Output dim: 13, lower bound: -115.7641302, upper bound: 115.9283281
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 211.75
Output dim: 13, lower bound: -115.9283281, upper bound: 115.7641302
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 211.75
Output dim: 13, lower bound: -115.8852544, upper bound: 115.8071756

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1630

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8043854, upper bound: 115.8028429
time: 113.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7254193, upper bound: 115.8825062
time: 131.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1630

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7613755, upper bound: 115.8465400
time: 97.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.6817848, upper bound: 115.9255515
time: 103.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1630

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.9255515, upper bound: 115.6817848
time: 492.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8465400, upper bound: 115.7613755
time: 97.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1630

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8825062, upper bound: 115.7254193
time: 98.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8028429, upper bound: 115.8043854
time: 127.62 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 231.75 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 231.75
Output dim: 13, lower bound: -115.8043854, upper bound: 115.8028429
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 231.75
Output dim: 13, lower bound: -115.7254193, upper bound: 115.8825062
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 231.75
Output dim: 13, lower bound: -115.7613755, upper bound: 115.8465400
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 231.75
Output dim: 13, lower bound: -115.6817848, upper bound: 115.9255515
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 231.75
Output dim: 13, lower bound: -115.9255515, upper bound: 115.6817848
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 231.75
Output dim: 13, lower bound: -115.8465400, upper bound: 115.7613755
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 231.75
Output dim: 13, lower bound: -115.8825062, upper bound: 115.7254193
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 231.75
Output dim: 13, lower bound: -115.8028429, upper bound: 115.8043854

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7053452, upper bound: 115.7982708
time: 131.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7997701, upper bound: 115.7076466
time: 110.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.6261326, upper bound: 115.8780104
time: 563.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7206154, upper bound: 115.7873275
time: 97.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.6622666, upper bound: 115.8420294
time: 97.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7566978, upper bound: 115.7513567
time: 195.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.5824908, upper bound: 115.9211502
time: 101.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.6768877, upper bound: 115.8303681
time: 172.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8303681, upper bound: 115.6768877
time: 1072.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.9211502, upper bound: 115.5824908
time: 92.59 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 1171.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1171.51
Output dim: 13, lower bound: -115.7053452, upper bound: 115.7982708
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1171.51
Output dim: 13, lower bound: -115.7997701, upper bound: 115.7076466
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1171.51
Output dim: 13, lower bound: -115.6261326, upper bound: 115.8780104
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1171.51
Output dim: 13, lower bound: -115.7206154, upper bound: 115.7873275
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1171.51
Output dim: 13, lower bound: -115.6622666, upper bound: 115.8420294
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1171.51
Output dim: 13, lower bound: -115.7566978, upper bound: 115.7513567
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1171.51
Output dim: 13, lower bound: -115.5824908, upper bound: 115.9211502
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1171.51
Output dim: 13, lower bound: -115.6768877, upper bound: 115.8303681
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1171.51
Output dim: 13, lower bound: -115.8303681, upper bound: 115.6768877
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1171.51
Output dim: 13, lower bound: -115.9211502, upper bound: 115.5824908
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1171.51
Output dim: 13, lower bound: -115.8465400, upper bound: 115.7613755
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1171.51
Output dim: 13, lower bound: -115.8825062, upper bound: 115.7254193
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1171.51
Output dim: 13, lower bound: -115.8028429, upper bound: 115.8043854
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=172.8270263671875
rel_dist={13: [-115.95178022674816, 115.95178022664516]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1725

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9242644, upper bound: 114.0376414
time: 89.15 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0376414, upper bound: 113.9242644
time: 114.65 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 203.96 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 203.96
Output dim: 13, lower bound: -113.9242644, upper bound: 114.0376414
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 203.96
Output dim: 13, lower bound: -114.0376414, upper bound: 113.9242644

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1597

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9202556, upper bound: 113.9926853
time: 101.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.8792583, upper bound: 114.0336366
time: 144.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1597

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0336366, upper bound: 113.8792583
time: 109.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9926853, upper bound: 113.9202556
time: 98.02 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 209.57 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 209.57
Output dim: 13, lower bound: -113.9202556, upper bound: 113.9926853
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 209.57
Output dim: 13, lower bound: -113.8792583, upper bound: 114.0336366
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 209.57
Output dim: 13, lower bound: -114.0336366, upper bound: 113.8792583
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 209.57
Output dim: 13, lower bound: -113.9926853, upper bound: 113.9202556

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1630

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.8768177, upper bound: 113.9186210
time: 97.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.8460850, upper bound: 113.9902519
time: 127.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1630

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.8768177, upper bound: 113.9597221
time: 156.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.8049276, upper bound: 114.0312083
time: 108.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1630

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0312083, upper bound: 113.8049276
time: 150.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9597221, upper bound: 113.8768177
time: 105.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1630

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9902520, upper bound: 113.8460849
time: 83.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.9186210, upper bound: 113.9178223
time: 95.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 185.17 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 185.17
Output dim: 13, lower bound: -113.8768177, upper bound: 113.9186210
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 185.17
Output dim: 13, lower bound: -113.8460850, upper bound: 113.9902519
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 185.17
Output dim: 13, lower bound: -113.8768177, upper bound: 113.9597221
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 185.17
Output dim: 13, lower bound: -113.8049276, upper bound: 114.0312083
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 185.17
Output dim: 13, lower bound: -114.0312083, upper bound: 113.8049276
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 185.17
Output dim: 13, lower bound: -113.9597221, upper bound: 113.8768177
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 185.17
Output dim: 13, lower bound: -113.9902520, upper bound: 113.8460849
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 185.17
Output dim: 13, lower bound: -113.9186210, upper bound: 113.9178223

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.7517279, upper bound: 113.9856128
time: 98.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.8412530, upper bound: 113.8990349
time: 125.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.7826103, upper bound: 113.9550853
time: 138.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.8720071, upper bound: 113.8683326
time: 124.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.7104995, upper bound: 114.0265715
time: 99.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.8000593, upper bound: 113.9400119
time: 125.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.9400119, upper bound: 113.8000593
time: 99.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0265715, upper bound: 113.7104994
time: 95.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.8271959, upper bound: 113.8720071
time: 98.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9550853, upper bound: 113.7826102
time: 141.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.8990349, upper bound: 113.8412530
time: 100.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9856128, upper bound: 113.7517279
time: 135.38 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 242.39 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 242.39
Output dim: 13, lower bound: -113.7517279, upper bound: 113.9856128
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 242.39
Output dim: 13, lower bound: -113.8412530, upper bound: 113.8990349
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 242.39
Output dim: 13, lower bound: -113.7826103, upper bound: 113.9550853
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 242.39
Output dim: 13, lower bound: -113.8720071, upper bound: 113.8683326
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 242.39
Output dim: 13, lower bound: -113.7104995, upper bound: 114.0265715
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 242.39
Output dim: 13, lower bound: -113.8000593, upper bound: 113.9400119
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 242.39
Output dim: 13, lower bound: -113.9400119, upper bound: 113.8000593
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 242.39
Output dim: 13, lower bound: -114.0265715, upper bound: 113.7104994
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 242.39
Output dim: 13, lower bound: -113.8271959, upper bound: 113.8720071
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 242.39
Output dim: 13, lower bound: -113.9550853, upper bound: 113.7826102
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 242.39
Output dim: 13, lower bound: -113.8990349, upper bound: 113.8412530
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 242.39
Output dim: 13, lower bound: -113.9856128, upper bound: 113.7517279

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.6341282, upper bound: 113.9822211
time: 104.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.7481338, upper bound: 113.8771697
time: 159.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.6653323, upper bound: 113.9516859
time: 101.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.7790105, upper bound: 113.8464264
time: 98.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.5927985, upper bound: 114.0231795
time: 87.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.7069057, upper bound: 113.9181578
time: 160.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 13, lower bound: -113.9181578, upper bound: 113.7069057
time: 110.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0231795, upper bound: 113.5927985
time: 116.77 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 233.65 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 233.65
Output dim: 13, lower bound: -113.6341282, upper bound: 113.9822211
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 233.65
Output dim: 13, lower bound: -113.7481338, upper bound: 113.8771697
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 233.65
Output dim: 13, lower bound: -113.6653323, upper bound: 113.9516859
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 233.65
Output dim: 13, lower bound: -113.7790105, upper bound: 113.8464264
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 233.65
Output dim: 13, lower bound: -113.5927985, upper bound: 114.0231795
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 233.65
Output dim: 13, lower bound: -113.7069057, upper bound: 113.9181578
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 233.65
Output dim: 13, lower bound: -113.9181578, upper bound: 113.7069057
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 233.65
Output dim: 13, lower bound: -114.0231795, upper bound: 113.5927985
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 233.65
Output dim: 13, lower bound: -113.9550853, upper bound: 113.7826102
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 233.65
Output dim: 13, lower bound: -113.9856128, upper bound: 113.7517279
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=172.8270263671875
rel_dist={13: [-114.05422291258427, 114.05422290962252]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 12865.53 seconds
