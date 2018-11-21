# -*- coding: UTF-8 -*-
"""
@author: Luca Bondi (luca.bondi@polimi.it)
@author: Paolo Bestagini (paolo.bestagini@polimi.it)
@author: Nicolò Bonettini (nicolo.bonettini@polimi.it)
"""
import os
import unittest
from glob import glob
from multiprocessing import cpu_count, Pool

import numpy as np
import torch
from PIL import Image

from prnu import ArgumentError, cut_ctr

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


class TestPrnu(unittest.TestCase):

    def test_init(self):
        self.assertRaises(ArgumentError, Prnu, mode='bm3d')

    def test_check(self):
        prnu = Prnu()
        self.assertRaises(ArgumentError, prnu.check, np.zeros((100, 200, 3)))
        self.assertRaises(ArgumentError, prnu.check, np.zeros((100, 200, 1)))
        self.assertRaises(ArgumentError, prnu.check, np.zeros((100, 200)))
        self.assertTrue(prnu.check(np.zeros((100, 200), np.uint8)))
        self.assertTrue(prnu.check(np.zeros((100, 200, 3), np.uint8)))

    def test_extract(self):
        prnu = Prnu()

        im1 = np.array(Image.open('data/camera.jpg'))[:400, :500]

        w = prnu.extract(im1)

        self.assertSequenceEqual(w.shape, im1.shape[:2])

    def test_extract_multiple(self):

        prnu = Prnu()

        im1 = np.asarray(Image.open('data/prnu1.jpg'))[:400, :500]
        im2 = np.asarray(Image.open('data/prnu2.jpg'))[:400, :500]

        imgs = [im1, im2]

        k_st = prnu.extract_multiple(imgs, processes=1)
        k_mt = prnu.extract_multiple(imgs, processes=2)

        self.assertTrue(np.allclose(k_st, k_mt, atol=1e-6))

    def test_crosscorr2d(self):
        prnu = Prnu()
        im = np.asarray(Image.open('data/prnu1.jpg'))[:1000, :800]

        w_all = prnu.extract(im)

        y_os, x_os = 300, 150
        w_cut = w_all[y_os:, x_os:]

        cc = prnu.crosscorr2d(w_cut, w_all)

        max_idx = np.argmax(cc.flatten())
        max_y, max_x = np.unravel_index(max_idx, cc.shape)

        peak_y = cc.shape[0] - 1 - max_y
        peak_x = cc.shape[1] - 1 - max_x

        peak_height = cc[max_y, max_x]

        self.assertSequenceEqual((peak_y, peak_x), (y_os, x_os))
        self.assertTrue(np.allclose(peak_height, 662901.0))

    def test_crosscorr2d_cuda(self):

        if not torch.cuda.is_available():
            print('Can\'t test when CUDA is not available')
            return

        prnu = Prnu()
        im1 = np.asarray(Image.open('data/prnu1.jpg'))[:1000, :800]
        im2 = np.asarray(Image.open('data/prnu2.jpg'))[:1000, :800]

        w_all1 = prnu.extract(im1)
        w_all2 = prnu.extract(im2)

        y_os, x_os = 300, 150
        w_cut1 = w_all1[y_os:, x_os:]
        w_cut2 = w_all2[y_os:, x_os:]

        batch_w_all = np.stack([w_all1, w_all2], axis=2)
        batch_w_cut = np.stack([w_cut1, w_cut2], axis=2)

        batch_cc = prnu.crosscorr2d_cuda(batch_w_cut, batch_w_all)
        cc1 = batch_cc[:, :, 0]
        cc2 = batch_cc[:, :, 1]

        max_idx1 = np.argmax(cc1.flatten())
        max_y1, max_x1 = np.unravel_index(max_idx1, cc1.shape)
        max_idx2 = np.argmax(cc2.flatten())
        max_y2, max_x2 = np.unravel_index(max_idx2, cc2.shape)

        peak_y1 = cc1.shape[0] - 1 - max_y1
        peak_x1 = cc1.shape[1] - 1 - max_x1
        peak_y2 = cc2.shape[0] - 1 - max_y2
        peak_x2 = cc2.shape[1] - 1 - max_x2

        peak_height1 = cc1[max_y1, max_x1]
        peak_height2 = cc2[max_y2, max_x2]

        cc1_np = prnu.crosscorr2d(w_cut1, w_all1)

        self.assertSequenceEqual((peak_y1, peak_x1, peak_y2, peak_x2), (y_os, x_os, y_os, x_os))
        self.assertTrue(np.allclose(peak_height1, 590507.9))
        self.assertTrue(np.allclose(peak_height2, 651970.44))
        self.assertTrue(np.allclose(cc1_np, cc1, atol=0.045))

    def test_pce(self):

        prnu = Prnu()
        im = np.asarray(Image.open('data/prnu1.jpg'))[:500, :400]

        w_all = prnu.extract(im)

        y_os, x_os = 5, 8
        w_cut = w_all[y_os:, x_os:]

        cc1 = prnu.crosscorr2d(w_cut, w_all)
        cc2 = prnu.crosscorr2d(w_all, w_cut)

        pce1 = prnu.pce(cc1)
        pce2 = prnu.pce(cc2)

        self.assertSequenceEqual(pce1['peak'], (im.shape[0] - y_os - 1, im.shape[1] - x_os - 1))
        self.assertTrue(np.allclose(pce1['pce'], 134791.14398835122))

        self.assertSequenceEqual(pce2['peak'], (y_os - 1, x_os - 1))
        self.assertTrue(np.allclose(pce2['pce'], 134797.4600680655))

    def test_detection(self):

        prnu = Prnu()

        nat = np.asarray(Image.open('data/prnu/nat-jpg/Nikon_D70s_0_22115.JPG'))
        ff1 = np.asarray(Image.open('data/prnu/ff-jpg/Nikon_D70s_0_22193.JPG'))
        ff2 = np.asarray(Image.open('data/prnu/ff-jpg/Nikon_D70s_1_23220.JPG'))

        nat = cut_ctr(nat, (500, 500, 3))
        ff1 = cut_ctr(ff1, (500, 500, 3))
        ff2 = cut_ctr(ff2, (500, 500, 3))

        w = prnu.extract(nat)
        k1 = prnu.extract(ff1)
        k2 = prnu.extract(ff2)

        pce1 = [{}] * 4
        pce2 = [{}] * 4

        for rot_idx in range(4):
            cc1 = prnu.crosscorr2d(k1, np.rot90(w, rot_idx))
            pce1[rot_idx] = prnu.pce(cc1)

            cc2 = prnu.crosscorr2d(k2, np.rot90(w, rot_idx))
            pce2[rot_idx] = prnu.pce(cc2)

        best_pce1 = np.max([p['pce'] for p in pce1])
        best_pce2 = np.max([p['pce'] for p in pce2])

        self.assertGreater(best_pce1, best_pce2)

    def test_gt(self):

        cams = ['a', 'b', 'c', 'd']
        nat = ['a', 'a', 'b', 'b', 'c', 'c', 'c']

        prnu = Prnu()

        gt = prnu.gt(cams, nat)

        self.assertTrue(np.allclose(gt, [[1, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, ], [0, 0, 0, 0, 1, 1, 1],
                                         [0, 0, 0, 0, 0, 0, 0, ]]))

    def test_stats(self):

        prnu = Prnu()

        gt = np.array([[1, 0, 0, ], [0, 1, 0], [0, 0, 1]], np.bool)
        cc = np.array([[0.5, 0.2, 0.1], [0.1, 0.7, 0.1], [0.4, 0.3, 0.9]])

        stats = prnu.stats(cc, gt)

        self.assertTrue(np.allclose(stats['auc'], 1))
        self.assertTrue(np.allclose(stats['eer'], 0))
        self.assertTrue(np.allclose(stats['tpr'][-1], 1))
        self.assertTrue(np.allclose(stats['fpr'][-1], 1))
        self.assertTrue(np.allclose(stats['tpr'][0], 0))
        self.assertTrue(np.allclose(stats['fpr'][0], 0))

    def test_detection_large(self):

        prnu = Prnu()

        ff_dirlist = np.array(sorted(glob('data/prnu/ff-jpg/*.JPG')))
        ff_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in ff_dirlist])

        nat_dirlist = np.array(sorted(glob('data/prnu/nat-jpg/*.JPG')))
        nat_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in nat_dirlist])

        print('Computing fingerprints')
        fingerprint_device = sorted(np.unique(ff_device))
        k = []
        for device in fingerprint_device:
            imgs = []
            for img_path in ff_dirlist[ff_device == device]:
                im = Image.open(img_path)
                im_arr = np.asarray(im)
                im_cut = cut_ctr(im_arr, (512, 512, 3))
                imgs += [im_cut]
            k += [prnu.extract_multiple(imgs, processes=cpu_count())]

        k = np.stack(k, 0)

        print('Computing residuals')

        imgs = []
        for img_path in nat_dirlist:
            imgs += [cut_ctr(np.asarray(Image.open(img_path)), (512, 512, 3))]

        pool = Pool(cpu_count())
        w = pool.map(prnu.extract, imgs)
        pool.close()

        w = np.stack(w, 0)

        # Computing Ground Truth
        gt = prnu.gt(fingerprint_device, nat_device)

        print('Computing rotation tolerant aligned cross correlation')
        # Suppose aligned, rotation unknown
        cc_aligned_rot_max = np.stack(
            [prnu.aligned_cc(k, np.rot90(w, rot_idx, axes=(1, 2)))['cc'] for rot_idx in range(4)]).max(0)

        print('Computing statistics on rotation tolerant aligned cross correlation')
        stats_cc = prnu.stats(cc_aligned_rot_max, gt)

        print('Computing rotation tolerant PCE')
        pce = np.zeros((4, len(fingerprint_device), len(nat_device)))

        for fingerprint_idx, fingerprint_k in enumerate(k):
            for natural_idx, natural_w in enumerate(w):
                for rot_idx in range(4):
                    cc2d = prnu.crosscorr2d(fingerprint_k, np.rot90(natural_w, rot_idx))
                    pce[rot_idx, fingerprint_idx, natural_idx] = prnu.pce(cc2d)['pce']
        pce_rot_max = pce.max(axis=0)

        print('Computing statistics on rotation PCE')
        stats_pce = prnu.stats(pce_rot_max, gt)

        self.assertTrue(np.allclose(stats_cc['auc'], 0.96349996))

        self.assertTrue(np.allclose(stats_pce['auc'], 0.85825002))


if __name__ == '__main__':
    unittest.main()
