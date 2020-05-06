"""
GUI to visualize emittance meter slit scans

Created 2020-05-05

@author: Filip Lindau

"""

import numpy as np
import pyqtgraph as pq
from scipy.signal import medfilt2d, medfilt
import sys
import os
from PyQt5 import QtWidgets, QtCore
from analysis_viewer_ui import Ui_Dialog
import logging

logger = logging.getLogger()
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


class EmittanceMeterViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self.image_width = 240
        self.image_height = 240
        self.px = 5.0 / self.image_width

        self.file_model = QtWidgets.QFileSystemModel()
        self.file_model.setFilter(QtCore.QDir.NoDotAndDotDot | QtCore.QDir.Files)
        # self.path = QtCore.QDir("./data")
        self.path = os.path.join(os.path.curdir, "data")
        self.file_model.setRootPath(self.path)

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setup_layout()

    def setup_layout(self):
        self.ui.file_listview.setModel(self.file_model)
        self.ui.file_listview.setRootIndex(self.file_model.index(self.path))
        self.ui.file_listview.activated.connect(self.process_image)
        self.ui.file_listview.clicked.connect(self.process_image)
        self.ui.processed_radiobutton.toggled.connect(self.process_image)
        self.ui.tight_radiobutton.toggled.connect(self.process_image)
        self.ui.bkg_spinbox.editingFinished.connect(self.process_image)
        self.ui.roi_top_spinbox.editingFinished.connect(self.process_image)
        self.ui.roi_height_spinbox.editingFinished.connect(self.process_image)
        self.ui.roi_left_spinbox.editingFinished.connect(self.process_image)
        self.ui.roi_width_spinbox.editingFinished.connect(self.process_image)
        self.ui.medfilt_spinbox.editingFinished.connect(self.process_image)
        self.ui.mask_spinbox.editingFinished.connect(self.process_image)

    def process_image2(self, index=None):
        if index is None:
            index = self.ui.file_listview.selectedIndexes()
        try:
            logger.info("Processing image {0}, {1}".format(index.row(), index.data()))
        except AttributeError:
            il = self.ui.file_listview.selectedIndexes()
            if len(il) > 0:
                index = self.ui.file_listview.selectedIndexes()[0]
            else:
                self.ui.image_size_label.setText("ERROR")
                self.ui.moment_1_label.setText("-.--")
                self.ui.moment_2_label.setText("-.--")
                return
        try:
            pic = np.load("{0}/data/{1}".format(QtCore.QDir.currentPath(), index.data()))
        except ValueError:
            self.ui.image_size_label.setText("ERROR")
            self.ui.moment_1_label.setText("-.--")
            self.ui.moment_2_label.setText("-.--")
            return
        self.ui.image_size_label.setText("{0} x {1}".format(pic.shape[1], pic.shape[0]))
        # logger.debug("Image size: {0} x {1}".format(pic.shape[0], pic.shape[1]))
        roi_t = self.ui.roi_top_spinbox.value()
        roi_h = self.ui.roi_height_spinbox.value()
        roi_l = self.ui.roi_left_spinbox.value()
        roi_w = self.ui.roi_width_spinbox.value()
        kernel = self.ui.medfilt_spinbox.value()
        if kernel % 2 == 0:
            kernel += 1
        pic_roi = np.double(pic[roi_t:roi_t + roi_h, roi_l:roi_l + roi_w])
        pic_proc = medfilt2d(np.double(pic[roi_t:roi_t+roi_h, roi_l:roi_l+roi_w]), kernel)
        bkg_level = pic_proc[:, 0:10].mean()
        self.ui.bkg_label.setText("{0:.1f}".format(bkg_level))
        bkg_cut = self.ui.bkg_spinbox.value()
        pic_proc = np.maximum(0, pic_proc - 1.5 * bkg_level)

        dx = 13e-6
        dx = 1
        x = dx * np.arange(pic_proc.shape[1])
        x0 = ((pic_proc * x[np.newaxis, :]).sum(1) / pic_proc.sum(1))
        x1 = (pic_proc * (x[np.newaxis, :] - x0[:, np.newaxis]) ** 2).sum(1) / pic_proc.sum(1)
        x0_good = x0[~np.isnan(x0)]
        x1_good = x1[~np.isnan(x1)]
        xc = np.median(x0_good)
        xs = np.median(x1_good)
        logger.debug("x0: {0}, x1: {1}".format(x0_good.shape, x1_good.shape))
        logger.debug("xc: {0:.1f}, xs: {1:.1f}".format(xc, xs))
        pic_roi2 = pic_roi[:, int(xc - xs * 5):int(xc + xs * 5)]
        pic_proc2 = medfilt2d(np.double(pic_roi2), kernel)
        pic_proc2 = np.maximum(0, pic_proc2 - bkg_cut)
        xt = dx * np.arange(pic_proc2.shape[1])
        xt0 = ((pic_proc2 * xt[np.newaxis, :]).sum(1) / pic_proc2.sum(1))
        xt1 = (pic_proc2 * (xt[np.newaxis, :] - xt0[:, np.newaxis]) ** 2).sum(1) / pic_proc2.sum(1)
        xt0_good = xt0[~np.isnan(xt0)]
        xt1_good = xt1[~np.isnan(xt1)]
        self.ui.moment_1_label.setText("{0:.1f} pixels".format(xt0_good.mean()))
        self.ui.moment_2_label.setText("{0:.1f} pixels".format(xt1_good.mean()))
        if self.ui.processed_radiobutton.isChecked():
            self.ui.image_widget.setImage(pic_proc.transpose(), autoLevels=True)
        elif self.ui.tight_radiobutton.isChecked():
            self.ui.image_widget.setImage(pic_proc2.transpose(), autoLevels=True)
        else:
            self.ui.image_widget.setImage(pic_roi.transpose(), autoLevels=False)

    def process_image(self, index=None):
        if index is None:
            index = self.ui.file_listview.selectedIndexes()
        try:
            logger.info("Processing image {0}, {1}".format(index.row(), index.data()))
        except AttributeError:
            il = self.ui.file_listview.selectedIndexes()
            if len(il) > 0:
                index = self.ui.file_listview.selectedIndexes()[0]
            else:
                self.ui.image_size_label.setText("ERROR")
                self.ui.moment_1_label.setText("-.--")
                self.ui.moment_2_label.setText("-.--")
                return
        try:
            # Load selected image
            pic = np.load("{0}/data/{1}".format(QtCore.QDir.currentPath(), index.data()))
        except ValueError:
            self.ui.image_size_label.setText("ERROR")
            self.ui.moment_1_label.setText("-.--")
            self.ui.moment_2_label.setText("-.--")
            return
        self.ui.image_size_label.setText("{0} x {1}".format(pic.shape[1], pic.shape[0]))
        # logger.debug("Image size: {0} x {1}".format(pic.shape[0], pic.shape[1]))

        # Large ROI around spot, the spot should always be inside
        roi_t = self.ui.roi_top_spinbox.value()
        roi_h = self.ui.roi_height_spinbox.value()
        roi_l = self.ui.roi_left_spinbox.value()
        roi_w = self.ui.roi_width_spinbox.value()

        kernel = self.ui.medfilt_spinbox.value()
        if kernel % 2 == 0:
            kernel += 1

        pic_roi = np.double(pic[roi_t:roi_t + roi_h, roi_l:roi_l + roi_w])
        pic_proc = medfilt2d(np.double(pic[roi_t:roi_t+roi_h, roi_l:roi_l+roi_w]), kernel)
        # Background level from first 20 columns, one level for each row (the background structure is banded):
        bkg_level = pic_proc[:, 0:20].mean(1)
        self.ui.bkg_label.setText("{0:.1f}".format(bkg_level.mean()))
        pic_proc = np.maximum(0, pic_proc - 1.1 * bkg_level[:, np.newaxis])

        dx = 13e-6
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
        xs = 10
        logger.debug("x0: {0}, x1: {1}".format(x0_good.shape, x1_good.shape))
        logger.debug("xc: {0}, xs: {1}".format(xc, xs))
        # Cut an new ROI around the central pixel column xc, width xs * 10
        pic_roi2 = pic_roi[:, np.maximum(0, int(xc - xs * 5)):np.minimum(pic_roi.shape[1]-1, int(xc + xs * 5))]
        pic_proc2 = medfilt2d(np.double(pic_roi2), kernel)
        # Background level to cut in the reduced ROI:
        bkg_cut = self.ui.bkg_spinbox.value()
        pic_proc2 = np.maximum(0, pic_proc2 - bkg_cut)
        # Create mask around the signal spot by heavily median filtering in the vertical direction (mask_kern ~ 25)
        mask_kern = self.ui.mask_spinbox.value()
        if mask_kern % 2 == 0:
            mask_kern += 1
        mask = medfilt(np.maximum(0, pic_roi2 - bkg_cut), [mask_kern, kernel])
        pic_proc3 = pic_proc2 * (mask > 0)
        xt = dx * np.arange(pic_proc3.shape[1])
        xt0 = ((pic_proc3 * xt[np.newaxis, :]).sum(1) / pic_proc3.sum(1))
        xt1 = (pic_proc3 * (xt[np.newaxis, :] - xt0[:, np.newaxis]) ** 2).sum(1) / pic_proc3.sum(1)
        xt0_good = xt0[~np.isnan(xt0)]
        xt1_good = xt1[~np.isnan(xt1)]
        ind = ~np.isnan(xt1)
        charge = pic_proc3.sum()
        xt1_w = (xt1[ind] * pic_proc3[ind, :].sum(1)).sum() / charge
        self.ui.moment_1_label.setText("{0:.1f} pixels".format(xt1_w))
        self.ui.moment_2_label.setText("{0:.1f} pixels".format(xt1_good.mean()))
        self.ui.charge_label.setText("{0}".format(charge))
        if self.ui.processed_radiobutton.isChecked():
            self.ui.image_widget.setImage(pic_proc.transpose(), autoLevels=True)
        elif self.ui.tight_radiobutton.isChecked():
            try:
                self.ui.image_widget.setImage(pic_proc3.transpose(), autoLevels=True)
            except ValueError:
                logger.error("pic_proc3 shape {0}".format(pic_proc3.shape))
                self.ui.image_widget.setImage(pic_roi2.transpose(), autoLevels=True)
        else:
            self.ui.image_widget.setImage(pic_roi.transpose(), autoLevels=False)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = EmittanceMeterViewer()
    logger.info("EmittanceMeterViewer object created")
    myapp.show()
    logger.info("App show")
    sys.exit(app.exec_())
    logger.info("App exit")
