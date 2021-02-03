# -*- coding: UTF-8 -*-
"""
@author: Luca Bondi (luca.bondi@polimi.it)
@author: Paolo Bestagini (paolo.bestagini@polimi.it)
@author: Nicolò Bonettini (nicolo.bonettini@polimi.it)
Politecnico di Milano 2018
"""
import sys
sys.path.insert(0, '..')
import os
import unittest

import numpy as np
from PIL import Image

import prnu

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


class TestPrnu(unittest.TestCase):

    def test_extract(self):
        im1 = np.array(Image.open('data/camera.jpg'))[:400, :500]
        w = prnu.extract_single(im1)
        self.assertSequenceEqual(w.shape, im1.shape[:2])

    def test_extract_multiple(self):
        im1 = np.asarray(Image.open('data/prnu1.jpg'))[:400, :500]
        im2 = np.asarray(Image.open('data/prnu2.jpg'))[:400, :500]

        imgs = [im1, im2]

        k_st = prnu.extract_multiple_aligned(imgs, processes=1)
        k_mt = prnu.extract_multiple_aligned(imgs, processes=2)

        self.assertTrue(np.allclose(k_st, k_mt, atol=1e-6))

    def test_crosscorr2d(self):
        im = np.asarray(Image.open('data/prnu1.jpg'))[:1000, :800]

        w_all = prnu.extract_single(im)

        y_os, x_os = 300, 150
        w_cut = w_all[y_os:, x_os:]

        cc = prnu.crosscorr_2d(w_cut, w_all)

        max_idx = np.argmax(cc.flatten())
        max_y, max_x = np.unravel_index(max_idx, cc.shape)

        peak_y = cc.shape[0] - 1 - max_y
        peak_x = cc.shape[1] - 1 - max_x

        peak_height = cc[max_y, max_x]

        self.assertSequenceEqual((peak_y, peak_x), (y_os, x_os))
        self.assertTrue(np.allclose(peak_height, 666995.0))

    def test_pce(self):
        im = np.asarray(Image.open('data/prnu1.jpg'))[:500, :400]

        w_all = prnu.extract_single(im)

        y_os, x_os = 5, 8
        w_cut = w_all[y_os:, x_os:]

        cc1 = prnu.crosscorr_2d(w_cut, w_all)
        cc2 = prnu.crosscorr_2d(w_all, w_cut)

        pce1 = prnu.pce(cc1)
        pce2 = prnu.pce(cc2)

        self.assertSequenceEqual(pce1['peak'], (im.shape[0] - y_os - 1, im.shape[1] - x_os - 1))
        self.assertTrue(np.allclose(pce1['pce'], 134611.58644973233))

        self.assertSequenceEqual(pce2['peak'], (y_os - 1, x_os - 1))
        self.assertTrue(np.allclose(pce2['pce'], 134618.03404934643))

    def test_gt(self):
        cams = ['a', 'b', 'c', 'd']
        nat = ['a', 'a', 'b', 'b', 'c', 'c', 'c']

        gt = prnu.gt(cams, nat)

        self.assertTrue(np.allclose(gt, [[1, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, ], [0, 0, 0, 0, 1, 1, 1],
                                         [0, 0, 0, 0, 0, 0, 0, ]]))

    def test_stats(self):
        gt = np.array([[1, 0, 0, ], [0, 1, 0], [0, 0, 1]], np.bool)
        cc = np.array([[0.5, 0.2, 0.1], [0.1, 0.7, 0.1], [0.4, 0.3, 0.9]])

        stats = prnu.stats(cc, gt)

        self.assertTrue(np.allclose(stats['auc'], 1))
        self.assertTrue(np.allclose(stats['eer'], 0))
        self.assertTrue(np.allclose(stats['tpr'][-1], 1))
        self.assertTrue(np.allclose(stats['fpr'][-1], 1))
        self.assertTrue(np.allclose(stats['tpr'][0], 0))
        self.assertTrue(np.allclose(stats['fpr'][0], 0))

    def test_detection(self):
        nat = np.asarray(Image.open('data/nat-jpg/Nikon_D70s_0_22115.JPG'))
        ff1 = np.asarray(Image.open('data/ff-jpg/Nikon_D70s_0_22193.JPG'))
        ff2 = np.asarray(Image.open('data/ff-jpg/Nikon_D70s_1_23220.JPG'))

        nat = prnu.cut_ctr(nat, (500, 500, 3))
        ff1 = prnu.cut_ctr(ff1, (500, 500, 3))
        ff2 = prnu.cut_ctr(ff2, (500, 500, 3))

        w = prnu.extract_single(nat)
        k1 = prnu.extract_single(ff1)
        k2 = prnu.extract_single(ff2)

        pce1 = [{}] * 4
        pce2 = [{}] * 4

        for rot_idx in range(4):
            cc1 = prnu.crosscorr_2d(k1, np.rot90(w, rot_idx))
            pce1[rot_idx] = prnu.pce(cc1)

            cc2 = prnu.crosscorr_2d(k2, np.rot90(w, rot_idx))
            pce2[rot_idx] = prnu.pce(cc2)

        best_pce1 = np.max([p['pce'] for p in pce1])
        best_pce2 = np.max([p['pce'] for p in pce2])

        self.assertGreater(best_pce1, best_pce2)


if __name__ == '__main__':
    unittest.main()
