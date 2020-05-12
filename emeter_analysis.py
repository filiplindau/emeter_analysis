"""
Analysis of emittance meter slit scans

Created 2020-05-07

@author: Filip Lindau

"""

import numpy as np
from scipy.signal import medfilt2d, medfilt
from scipy import ndimage
import sys
import os
import glob
import time
import threading
import multiprocessing as mp
import logging

logger = logging.getLogger()
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.INFO)
logging.getLogger('parso.python.diff').disabled = True
logging.getLogger('parso.cache').disabled = True
logging.getLogger('parso.cache.pickle').disabled = True


def process_image(pic, slit_pos, parameter_dict):
    kernel = parameter_dict["kernel"]
    mask_kern = parameter_dict["mask_kernel"]

    # Background level to cut in the reduced ROI:
    bkg_cut = parameter_dict["bkg_cut"]

    rotation = parameter_dict["rotation"]
    px = parameter_dict["px"]
    dist = parameter_dict["dist"]

    pic_proc = medfilt2d(np.double(pic), kernel)
    t01 = time.time()
    # Background level from first 20 columns, one level for each row (the background structure is banded):
    bkg_level = pic_proc[:, 0:20].mean(1)
    logger.debug("{0:.1f}".format(bkg_level.mean()))
    if bkg_cut == "auto":
        bkg_cut = bkg_level.max() + 1
    pic_proc = np.maximum(0, pic_proc - 1.1 * bkg_level[:, np.newaxis])

    x = np.arange(pic_proc.shape[1])
    x0 = ((pic_proc * x[np.newaxis, :]).sum(1) / pic_proc.sum(1))
    x1 = (pic_proc * (x[np.newaxis, :] - x0[:, np.newaxis]) ** 2).sum(1) / pic_proc.sum(1)
    x0_good = x0[~np.isnan(x0)]
    x1_good = x1[~np.isnan(x1)]

    # Assume spot is in the central +-50 vertical pixels of the ROI. Get the index for the maximum value
    xc = pic_proc[pic_proc.shape[0] // 2 - 50:pic_proc.shape[0] // 2 + 50, :].sum(0).argmax()
    xs = 100
    logger.debug("x0: {0}, x1: {1}".format(x0_good.shape, x1_good.shape))
    logger.debug("xc: {0}, xs: {1}".format(xc, xs))
    # Cut an new ROI around the central pixel column xc, width xs
    pic_roi2 = np.zeros((pic.shape[0], xs + 1))
    low = np.maximum(0, int(xs // 2 - xc))
    high = np.minimum(-1, int(pic.shape[1] - xc) - xs // 2)
    logger.debug("low {0}, high {1}".format(low, high))
    pic_roi2[:, np.maximum(0, int(xs // 2 - xc)):np.minimum(-1, int(pic.shape[1] - xc) - xs // 2)] = \
        pic[:, np.maximum(0, int(xc - xs // 2)):np.minimum(pic.shape[1] - 1, int(xc + xs // 2))]
    pic_roi2 = pic_roi2[:, :-1]
    t0 = time.time()
    pic_proc2 = medfilt2d(np.double(pic_roi2), kernel)
    t1 = time.time()
    pic_proc2 = np.maximum(0, pic_proc2 - bkg_cut)
    # Create mask around the signal spot by heavily median filtering in the vertical direction (mask_kern ~ 25)

    mask = medfilt(np.maximum(0, pic_roi2 - bkg_cut), [mask_kern, kernel])
    t2 = time.time()
    pic_proc3 = pic_proc2 * (mask > 0)
    xt = px * np.arange(pic_proc3.shape[1])
    xt0 = ((pic_proc3 * xt[np.newaxis, :]).sum(1) / pic_proc3.sum(1))
    xt1 = (pic_proc3 * (xt[np.newaxis, :] - xt0[:, np.newaxis]) ** 2).sum(1) / pic_proc3.sum(1)
    ind = ~np.isnan(xt1)
    charge = pic_proc3.sum()
    xt1_w = (xt1[ind] * pic_proc3[ind, :].sum(1)).sum() / charge
    logger.debug("Spot sigma: {0:.1f} pixels".format(xt1_w))
    logger.debug("Charge: {0}".format(charge))
    pic_proc3 = ndimage.rotate(pic_proc3, rotation * 180 / np.pi, reshape=False)

    line = pic_proc3.sum(0)
    x = np.arange(line.shape[0]) * px
    x0 = (x * line).sum() / charge
    xp2 = np.sqrt(((x - x0) ** 2 * line).sum() / charge) / dist
    xp = (x0 + xc * px - slit_pos) / dist
    result = {"charge": charge, "xc": xc, "xp": xp, "xp2": xp2, "pic_proc": pic_proc3, "slit_pos": slit_pos}
    return result


class EmittanceMeterAnalysis(object):
    def __init__(self):
        self.path = os.path.join(os.path.curdir, "data")
        self.roi_t = 250
        self.roi_h = 700
        self.roi_l = 0
        self.roi_w = 1400

        self.kernel = 5
        self.bkg_cut = "auto"
        self.mask_kernel = 25
        self.rotation = 0.0544

        self.dist = 0.23
        self.px = 13.3e-6

        self.image_data = list()
        self.pos_data = None
        self.charge_data = None
        self.image_center_data = None
        self.xp_data = None
        self.xp2_data = None

        self.pos_u = None
        self.ch_u = None
        self.center_u = None
        self.xp_u = None
        self.xp2_u = None

        self.x_e = None
        self.x2_e = None
        self.xp_e = None
        self.xp2_e = None
        self.xxp_e = None
        self.eps = None

        self.param_lock = threading.Lock()
        self.data_lock = threading.Lock()

    def analyze_scan(self, filename, bkg_cut=7):
        t_start = time.time()
        full_name = os.path.join(self.path, "{0}-*.npy".format(filename))
        file_list = glob.glob(full_name)
        logger.info("Looking for {0}. Found {1} files".format(full_name, len(file_list)))
        image_data = list()
        image_center_data = list()
        charge_data = list()
        pos_data = list()
        xp_data = list()
        xp2_data = list()

        self.bkg_cut = bkg_cut

        for ind, file in enumerate(file_list):
            t0 = time.time()
            logger.debug("\n===============================================================\n\n"
                         "  Processing image {0}: {1}/{2}\n".format(file, ind, len(file_list)))
            if not "bkg" in file:
                t1 = time.time()
                pos = np.double(file.split("-")[-2]) * 1e-3
                try:
                    pic = np.load(file)
                except ValueError:
                    logger.error("File {0} corrupted. Skipping.".format(file))
                    continue

                t2 = time.time()
                pp0, pp1, xc = self.process_image(pic)
                charge, xp, xp2 = self.analyse_image(pp1, pos, xc)
                t3 = time.time()
                pos_data.append(pos)
                image_data.append(pp1)
                image_center_data.append(xc)
                charge_data.append(charge)
                xp_data.append(xp)
                xp2_data.append(xp2)
            logger.debug("\n\nTotal time spent: {0:.1f} ms\n"
                         "Load time: {1:.1f} ms,\n"
                         "Process time: {2:.1f} ms\n".format(1e3*(time.time() - t0),
                                                             1e3*(t2-t1),
                                                             1e3*(t3-t2)))
        ind_sort = np.argsort(np.array(pos_data))
        ind_good = ~np.isnan(np.array(xp2_data)[ind_sort])
        self.image_data = image_data
        self.pos_data = np.array(pos_data)[ind_sort][ind_good]
        self.charge_data = np.array(charge_data)[ind_sort][ind_good]
        self.image_center_data = np.array(image_center_data)[ind_sort][ind_good]
        self.xp_data = np.array(xp_data)[ind_sort][ind_good]
        self.xp2_data = np.array(xp2_data)[ind_sort][ind_good]
        p_u = np.unique(self.pos_data)
        xp_u = np.zeros_like(p_u)
        xp2_u = np.zeros_like(p_u)
        ch_u = np.zeros_like(p_u)
        cent_u = np.zeros_like(p_u)
        for ind, p in enumerate(p_u):
            p_ind = self.pos_data == p
            xp_u[ind] = self.xp_data[p_ind].mean()
            xp2_u[ind] = self.xp2_data[p_ind].mean()
            ch_u[ind] = self.charge_data[p_ind].mean()
            cent_u[ind] = self.image_center_data[p_ind].mean() * self.px
        self.pos_u = p_u
        self.ch_u = ch_u
        self.center_u = cent_u
        self.xp_u = xp_u
        self.xp2_u = xp2_u

        ch_tot = ch_u.sum()
        self.x_e = (cent_u * ch_u).sum() / ch_tot
        self.x2_e = ((cent_u - self.x_e)**2 * ch_u).sum() / ch_tot
        self.xp_e = (xp_u * ch_u).sum() / ch_tot
        self.xp2_e = ((xp2_u**2 + (xp_u - self.xp_e)**2) * ch_u).sum() / ch_tot
        self.xxp_e = ((cent_u * xp_u * ch_u).sum() - ch_tot * self.x_e * self.xp_e) / ch_tot

        self.eps = np.sqrt(self.x2_e * self.xp2_e - self.xxp_e**2)

        logger.info("Total time: {0}".format(time.time() - t_start))

        return self.eps

    def analyze_scan_mp(self, filename):
        t0 = time.time()
        full_name = os.path.join(self.path, "{0}-*.npy".format(filename))
        file_list = glob.glob(full_name)
        logger.info("Looking for {0}. Found {1} files".format(full_name, len(file_list)))
        with self.data_lock:
            self.image_data = list()
            self.image_center_data = list()
            self.charge_data = list()
            self.pos_data = list()
            self.xp_data = list()
            self.xp2_data = list()

        parameter_dict = dict()

        with self.param_lock:
            parameter_dict["rotation"] = self.rotation
            parameter_dict["kernel"] = self.kernel
            parameter_dict["mask_kernel"] = self.mask_kernel
            parameter_dict["bkg_cut"] = self.bkg_cut
            parameter_dict["px"] = self.px
            parameter_dict["dist"] = self.dist
            roi_t = self.roi_t
            roi_h = self.roi_h
            roi_l = self.roi_l
            roi_w = self.roi_w

        pool = mp.Pool(mp.cpu_count())

        for ind, file in enumerate(file_list):
            t0 = time.time()
            logger.debug("\n===============================================================\n\n"
                         "  Processing image {0}: {1}/{2}\n".format(file, ind, len(file_list)))
            if not "bkg" in file:
                t1 = time.time()
                pos = np.double(file.split("-")[-2]) * 1e-3
                try:
                    pic = np.load(file)
                except ValueError:
                    logger.error("File {0} corrupted. Skipping.".format(file))
                    continue

                pic_roi = np.double(pic[roi_t:roi_t + roi_h, roi_l:roi_l + roi_w])
                pool.apply_async(process_image, (pic_roi, pos, parameter_dict), callback=self.result_callback)
        pool.close()
        pool.join()
        with self.data_lock:
            ind_sort = np.argsort(np.array(self.pos_data))
            ind_good = ~np.isnan(np.array(self.xp2_data)[ind_sort])
            self.pos_data = np.array(self.pos_data)[ind_sort][ind_good]
            self.charge_data = np.array(self.charge_data)[ind_sort][ind_good]
            self.image_center_data = np.array(self.image_center_data)[ind_sort][ind_good]
            self.xp_data = np.array(self.xp_data)[ind_sort][ind_good]
            self.xp2_data = np.array(self.xp2_data)[ind_sort][ind_good]

        eps = self.calc_emittance()
        logger.info("Total time: {0}".format(time.time() - t0))
        return eps

    def result_callback(self, res):
        logger.debug("Result: {0}".format(res))
        with self.data_lock:
            self.pos_data.append(res["slit_pos"])
            self.charge_data.append(res["charge"])
            self.xp_data.append(res["xp"])
            self.xp2_data.append(res["xp2"])
            self.image_data.append(res["pic_proc"])
            self.image_center_data.append(res["xc"])

    def calc_emittance(self):
        with self.data_lock:
            p_u = np.unique(self.pos_data)
            xp_u = np.zeros_like(p_u)
            xp2_u = np.zeros_like(p_u)
            ch_u = np.zeros_like(p_u)
            cent_u = np.zeros_like(p_u)
            for ind, p in enumerate(p_u):
                p_ind = self.pos_data == p
                xp_u[ind] = self.xp_data[p_ind].mean()
                xp2_u[ind] = self.xp2_data[p_ind].mean()
                ch_u[ind] = self.charge_data[p_ind].mean()
                cent_u[ind] = self.image_center_data[p_ind].mean() * self.px
        self.pos_u = p_u
        self.ch_u = ch_u
        self.center_u = cent_u
        self.xp_u = xp_u
        self.xp2_u = xp2_u

        ch_tot = ch_u.sum()
        self.x_e = (cent_u * ch_u).sum() / ch_tot
        self.x2_e = ((cent_u - self.x_e)**2 * ch_u).sum() / ch_tot
        self.xp_e = (xp_u * ch_u).sum() / ch_tot
        self.xp2_e = ((xp2_u**2 + (xp_u - self.xp_e)**2) * ch_u).sum() / ch_tot
        self.xxp_e = ((cent_u * xp_u * ch_u).sum() - ch_tot * self.x_e * self.xp_e) / ch_tot

        self.eps = np.sqrt(self.x2_e * self.xp2_e - self.xxp_e**2)

        return self.eps

    def process_image(self, pic, rotate=True):

        logger.debug("Image size: {0} x {1}".format(pic.shape[0], pic.shape[1]))

        t00 = time.time()
        # Large ROI around spot, the spot should always be inside
        with self.param_lock:
            roi_t = self.roi_t
            roi_h = self.roi_h
            roi_l = self.roi_l
            roi_w = self.roi_w

            kernel = self.kernel
            if kernel % 2 == 0:
                kernel += 1

            mask_kern = self.mask_kernel
            if mask_kern % 2 == 0:
                mask_kern += 1

            # Background level to cut in the reduced ROI:
            bkg_cut = self.bkg_cut

            rotation = self.rotation

        pic_roi = np.double(pic[roi_t:roi_t + roi_h, roi_l:roi_l + roi_w])
        pic_proc = medfilt2d(np.double(pic[roi_t:roi_t+roi_h, roi_l:roi_l+roi_w]), kernel)
        t01 = time.time()
        # Background level from first 20 columns, one level for each row (the background structure is banded):
        bkg_level = pic_proc[:, 0:20].mean(1)
        logger.debug("{0:.1f}".format(bkg_level.mean()))
        if self.bkg_cut == "auto":
            bkg_cut = bkg_level.max() + 1
        pic_proc = np.maximum(0, pic_proc - 1.1 * bkg_level[:, np.newaxis])

        dx = 1
        x = dx * np.arange(pic_proc.shape[1])
        x0 = ((pic_proc * x[np.newaxis, :]).sum(1) / pic_proc.sum(1))
        x1 = (pic_proc * (x[np.newaxis, :] - x0[:, np.newaxis]) ** 2).sum(1) / pic_proc.sum(1)
        x0_good = x0[~np.isnan(x0)]
        x1_good = x1[~np.isnan(x1)]
        # xc = np.median(x0_good)
        # xs = np.median(x1_good)

        # Assume spot is in the central +-50 vertical pixels of the ROI. Get the index for the maximum value
        xc = pic_proc[pic_proc.shape[0]//2 - 50:pic_proc.shape[0]//2 + 50, :].sum(0).argmax()
        xs = 100
        logger.debug("x0: {0}, x1: {1}".format(x0_good.shape, x1_good.shape))
        logger.debug("xc: {0}, xs: {1}".format(xc, xs))
        # Cut an new ROI around the central pixel column xc, width xs
        pic_roi2 = np.zeros((pic_roi.shape[0], xs + 1))
        low = np.maximum(0, int(xs // 2 - xc))
        high = np.minimum(-1, int(pic_roi.shape[1] - xc) - xs//2)
        logger.debug("low {0}, high {1}".format(low, high))
        pic_roi2[:, np.maximum(0, int(xs // 2 - xc)):np.minimum(-1, int(pic_roi.shape[1] - xc) - xs//2)] = \
            pic_roi[:, np.maximum(0, int(xc - xs // 2)):np.minimum(pic_roi.shape[1]-1, int(xc + xs // 2))]
        pic_roi2 = pic_roi2[:, :-1]
        t0 = time.time()
        pic_proc2 = medfilt2d(np.double(pic_roi2), kernel)
        t1 = time.time()
        pic_proc2 = np.maximum(0, pic_proc2 - bkg_cut)
        # Create mask around the signal spot by heavily median filtering in the vertical direction (mask_kern ~ 25)

        mask = medfilt(np.maximum(0, pic_roi2 - bkg_cut), [mask_kern, kernel])
        t2 = time.time()
        pic_proc3 = pic_proc2 * (mask > 0)
        xt = dx * np.arange(pic_proc3.shape[1])
        xt0 = ((pic_proc3 * xt[np.newaxis, :]).sum(1) / pic_proc3.sum(1))
        xt1 = (pic_proc3 * (xt[np.newaxis, :] - xt0[:, np.newaxis]) ** 2).sum(1) / pic_proc3.sum(1)
        ind = ~np.isnan(xt1)
        charge = pic_proc3.sum()
        xt1_w = (xt1[ind] * pic_proc3[ind, :].sum(1)).sum() / charge
        logger.debug("Spot sigma: {0:.1f} pixels".format(xt1_w))
        logger.debug("Charge: {0}".format(charge))
        if rotate:
            pic_proc3 = ndimage.rotate(pic_proc3, rotation * 180 / np.pi, reshape=False)
        t3 = time.time()
        logger.debug("\n\nMedfilt0: {3:.1f} ms\n"
                     "Medfilt2d: {0:.1f} ms, "
                     "\nmedfilt1: {1:.1f} ms, "
                     "\nrotate: {2:.1f} ms".format(1e3*(t1-t0),
                                                   1e3*(t2-t1),
                                                   1e3*(t3-t2),
                                                   1e3*(t01-t00)))
        return pic_proc, pic_proc3, xc

    def analyse_image(self, pic, slit_pos=None, pic_center=None):
        line = pic.sum(0)
        x = np.arange(line.shape[0]) * self.px
        charge = line.sum()
        x0 = (x * line).sum() / charge
        xp2 = np.sqrt(((x - x0)**2 * line).sum() / charge) / self.dist
        xp = (x0 + pic_center * self.px - slit_pos) / self.dist
        return charge, xp, xp2

    def set_roi(self, roi_t, roi_h, roi_l, roi_w):
        with self.param_lock:
            self.roi_t = roi_t
            self.roi_h = roi_h
            self.roi_l = roi_l
            self.roi_w = roi_w

    def set_rotation(self, rotation):
        with self.param_lock:
            self.rotation = rotation

    def set_bkg_cut(self, bkg_cut):
        with self.param_lock:
            self.bkg_cut = bkg_cut

    def set_median_kernels(self, kernel, mask_kernel):
        with self.param_lock:
            self.kernel = kernel
            self.mask_kernel = mask_kernel

    def set_parameters(self, parameter_dict):
        with self.param_lock:
            try:
                self.roi_t = parameter_dict["roi_t"]
                self.roi_h = parameter_dict["roi_h"]
                self.roi_l = parameter_dict["roi_l"]
                self.roi_w = parameter_dict["roi_w"]
            except KeyError:
                logger.error("No ROI data")
            try:
                self.rotation = parameter_dict["rotation"]
            except KeyError:
                logger.error("No rotation")
            try:
                self.bkg_cut = parameter_dict["bkg_cut"]
            except KeyError:
                logger.error("No bkg_cut")
            try:
                self.kernel = parameter_dict["kernel"]
                self.mask_kernel = parameter_dict["mask_kernel"]
            except KeyError:
                logger.error("No kernel")
            try:
                self.px = parameter_dict["px"]
                self.dist = parameter_dict["dist"]
            except KeyError:
                logger.error("Calibration")


if __name__ == "__main__":
    em = EmittanceMeterAnalysis()
