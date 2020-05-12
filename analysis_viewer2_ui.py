# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\analysis_viewer2_ui.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1100, 948)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Dialog)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(-1, -1, 6, -1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_16 = QtWidgets.QLabel(Dialog)
        self.label_16.setObjectName("label_16")
        self.verticalLayout.addWidget(self.label_16)
        self.dataset_treeview = QtWidgets.QTreeView(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.dataset_treeview.sizePolicy().hasHeightForWidth())
        self.dataset_treeview.setSizePolicy(sizePolicy)
        self.dataset_treeview.setObjectName("dataset_treeview")
        self.verticalLayout.addWidget(self.dataset_treeview)
        self.dataset_select_label = QtWidgets.QLabel(Dialog)
        self.dataset_select_label.setObjectName("dataset_select_label")
        self.verticalLayout.addWidget(self.dataset_select_label)
        self.file_listview = QtWidgets.QListView(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.file_listview.sizePolicy().hasHeightForWidth())
        self.file_listview.setSizePolicy(sizePolicy)
        self.file_listview.setObjectName("file_listview")
        self.verticalLayout.addWidget(self.file_listview)
        self.file_slider = QtWidgets.QSlider(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.file_slider.sizePolicy().hasHeightForWidth())
        self.file_slider.setSizePolicy(sizePolicy)
        self.file_slider.setOrientation(QtCore.Qt.Horizontal)
        self.file_slider.setObjectName("file_slider")
        self.verticalLayout.addWidget(self.file_slider)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setContentsMargins(-1, -1, -1, 6)
        self.gridLayout.setObjectName("gridLayout")
        self.beamenergy_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.beamenergy_spinbox.setMaximum(20.0)
        self.beamenergy_spinbox.setProperty("value", 3.3)
        self.beamenergy_spinbox.setObjectName("beamenergy_spinbox")
        self.gridLayout.addWidget(self.beamenergy_spinbox, 13, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.roi_width_spinbox = QtWidgets.QSpinBox(Dialog)
        self.roi_width_spinbox.setMaximum(3000)
        self.roi_width_spinbox.setProperty("value", 1400)
        self.roi_width_spinbox.setObjectName("roi_width_spinbox")
        self.gridLayout.addWidget(self.roi_width_spinbox, 4, 1, 1, 1)
        self.rotation_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.rotation_spinbox.setMinimum(-180.0)
        self.rotation_spinbox.setMaximum(180.0)
        self.rotation_spinbox.setSingleStep(0.5)
        self.rotation_spinbox.setProperty("value", 3.12)
        self.rotation_spinbox.setObjectName("rotation_spinbox")
        self.gridLayout.addWidget(self.rotation_spinbox, 6, 1, 1, 1)
        self.slit_screen_distance_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.slit_screen_distance_spinbox.setMaximum(200.0)
        self.slit_screen_distance_spinbox.setProperty("value", 23.0)
        self.slit_screen_distance_spinbox.setObjectName("slit_screen_distance_spinbox")
        self.gridLayout.addWidget(self.slit_screen_distance_spinbox, 12, 1, 1, 1)
        self.medfilt_spinbox = QtWidgets.QSpinBox(Dialog)
        self.medfilt_spinbox.setMinimum(1)
        self.medfilt_spinbox.setMaximum(13)
        self.medfilt_spinbox.setProperty("value", 5)
        self.medfilt_spinbox.setObjectName("medfilt_spinbox")
        self.gridLayout.addWidget(self.medfilt_spinbox, 7, 1, 1, 1)
        self.label_18 = QtWidgets.QLabel(Dialog)
        self.label_18.setObjectName("label_18")
        self.gridLayout.addWidget(self.label_18, 6, 0, 1, 1)
        self.bkg_spinbox = QtWidgets.QSpinBox(Dialog)
        self.bkg_spinbox.setMaximumSize(QtCore.QSize(16777209, 16777215))
        self.bkg_spinbox.setMaximum(65000)
        self.bkg_spinbox.setProperty("value", 7)
        self.bkg_spinbox.setObjectName("bkg_spinbox")
        self.gridLayout.addWidget(self.bkg_spinbox, 8, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(Dialog)
        self.label_12.setText("")
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 5, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem, 14, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(Dialog)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 8, 0, 1, 1)
        self.roi_left_spinbox = QtWidgets.QSpinBox(Dialog)
        self.roi_left_spinbox.setMaximum(3000)
        self.roi_left_spinbox.setObjectName("roi_left_spinbox")
        self.gridLayout.addWidget(self.roi_left_spinbox, 3, 1, 1, 1)
        self.pixelsize_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.pixelsize_spinbox.setSingleStep(1.0)
        self.pixelsize_spinbox.setProperty("value", 13.3)
        self.pixelsize_spinbox.setObjectName("pixelsize_spinbox")
        self.gridLayout.addWidget(self.pixelsize_spinbox, 11, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.label_14 = QtWidgets.QLabel(Dialog)
        self.label_14.setObjectName("label_14")
        self.gridLayout.addWidget(self.label_14, 9, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(Dialog)
        self.label_13.setText("")
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 10, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(Dialog)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 16, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 4, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(-1, -1, -1, 6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.raw_radiobutton = QtWidgets.QRadioButton(Dialog)
        self.raw_radiobutton.setObjectName("raw_radiobutton")
        self.horizontalLayout_2.addWidget(self.raw_radiobutton)
        self.processed_radiobutton = QtWidgets.QRadioButton(Dialog)
        self.processed_radiobutton.setChecked(True)
        self.processed_radiobutton.setObjectName("processed_radiobutton")
        self.horizontalLayout_2.addWidget(self.processed_radiobutton)
        self.tight_radiobutton = QtWidgets.QRadioButton(Dialog)
        self.tight_radiobutton.setObjectName("tight_radiobutton")
        self.horizontalLayout_2.addWidget(self.tight_radiobutton)
        self.gridLayout.addLayout(self.horizontalLayout_2, 16, 1, 1, 1)
        self.image_size_label = QtWidgets.QLabel(Dialog)
        self.image_size_label.setObjectName("image_size_label")
        self.gridLayout.addWidget(self.image_size_label, 0, 1, 1, 1)
        self.label_23 = QtWidgets.QLabel(Dialog)
        self.label_23.setObjectName("label_23")
        self.gridLayout.addWidget(self.label_23, 13, 0, 1, 1)
        self.label_22 = QtWidgets.QLabel(Dialog)
        self.label_22.setObjectName("label_22")
        self.gridLayout.addWidget(self.label_22, 12, 0, 1, 1)
        self.mask_spinbox = QtWidgets.QSpinBox(Dialog)
        self.mask_spinbox.setSingleStep(2)
        self.mask_spinbox.setProperty("value", 25)
        self.mask_spinbox.setObjectName("mask_spinbox")
        self.gridLayout.addWidget(self.mask_spinbox, 9, 1, 1, 1)
        self.roi_top_spinbox = QtWidgets.QSpinBox(Dialog)
        self.roi_top_spinbox.setMaximum(3000)
        self.roi_top_spinbox.setProperty("value", 250)
        self.roi_top_spinbox.setObjectName("roi_top_spinbox")
        self.gridLayout.addWidget(self.roi_top_spinbox, 1, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(Dialog)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 0, 0, 1, 1)
        self.roi_height_spinbox = QtWidgets.QSpinBox(Dialog)
        self.roi_height_spinbox.setMaximum(3000)
        self.roi_height_spinbox.setProperty("value", 700)
        self.roi_height_spinbox.setObjectName("roi_height_spinbox")
        self.gridLayout.addWidget(self.roi_height_spinbox, 2, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 7, 0, 1, 1)
        self.label_21 = QtWidgets.QLabel(Dialog)
        self.label_21.setObjectName("label_21")
        self.gridLayout.addWidget(self.label_21, 11, 0, 1, 1)
        self.label_26 = QtWidgets.QLabel(Dialog)
        self.label_26.setObjectName("label_26")
        self.gridLayout.addWidget(self.label_26, 17, 0, 1, 1)
        self.start_analysis_button = QtWidgets.QPushButton(Dialog)
        self.start_analysis_button.setObjectName("start_analysis_button")
        self.gridLayout.addWidget(self.start_analysis_button, 17, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(-1, -1, 6, -1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.image_widget = ImageView(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image_widget.sizePolicy().hasHeightForWidth())
        self.image_widget.setSizePolicy(sizePolicy)
        self.image_widget.setObjectName("image_widget")
        self.verticalLayout_2.addWidget(self.image_widget)
        self.line = QtWidgets.QFrame(Dialog)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_2.addWidget(self.line)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(-1, 6, -1, -1)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.plot_widget = PlotWidget(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_widget.sizePolicy().hasHeightForWidth())
        self.plot_widget.setSizePolicy(sizePolicy)
        self.plot_widget.setObjectName("plot_widget")
        self.horizontalLayout_3.addWidget(self.plot_widget)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setContentsMargins(6, -1, -1, -1)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_25 = QtWidgets.QLabel(Dialog)
        self.label_25.setObjectName("label_25")
        self.gridLayout_2.addWidget(self.label_25, 6, 0, 1, 1)
        self.charge_label = QtWidgets.QLabel(Dialog)
        self.charge_label.setObjectName("charge_label")
        self.gridLayout_2.addWidget(self.charge_label, 4, 1, 1, 1)
        self.moment_1_label = QtWidgets.QLabel(Dialog)
        self.moment_1_label.setObjectName("moment_1_label")
        self.gridLayout_2.addWidget(self.moment_1_label, 2, 1, 1, 1)
        self.eps_n_label = QtWidgets.QLabel(Dialog)
        self.eps_n_label.setObjectName("eps_n_label")
        self.gridLayout_2.addWidget(self.eps_n_label, 12, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 2, 0, 1, 1)
        self.label_15 = QtWidgets.QLabel(Dialog)
        self.label_15.setObjectName("label_15")
        self.gridLayout_2.addWidget(self.label_15, 4, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem1, 7, 0, 1, 1)
        self.label_19 = QtWidgets.QLabel(Dialog)
        self.label_19.setObjectName("label_19")
        self.gridLayout_2.addWidget(self.label_19, 9, 0, 1, 1)
        self.label_17 = QtWidgets.QLabel(Dialog)
        self.label_17.setText("")
        self.label_17.setObjectName("label_17")
        self.gridLayout_2.addWidget(self.label_17, 5, 0, 1, 1)
        self.label_24 = QtWidgets.QLabel(Dialog)
        self.label_24.setObjectName("label_24")
        self.gridLayout_2.addWidget(self.label_24, 11, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 3, 0, 1, 1)
        self.bkg_label = QtWidgets.QLabel(Dialog)
        self.bkg_label.setMinimumSize(QtCore.QSize(50, 0))
        self.bkg_label.setObjectName("bkg_label")
        self.gridLayout_2.addWidget(self.bkg_label, 1, 1, 1, 1)
        self.rejected_images_label = QtWidgets.QLabel(Dialog)
        self.rejected_images_label.setObjectName("rejected_images_label")
        self.gridLayout_2.addWidget(self.rejected_images_label, 6, 1, 1, 1)
        self.moment_2_label = QtWidgets.QLabel(Dialog)
        self.moment_2_label.setObjectName("moment_2_label")
        self.gridLayout_2.addWidget(self.moment_2_label, 3, 1, 1, 1)
        self.sigma_x_label = QtWidgets.QLabel(Dialog)
        self.sigma_x_label.setObjectName("sigma_x_label")
        self.gridLayout_2.addWidget(self.sigma_x_label, 9, 1, 1, 1)
        self.eps_label = QtWidgets.QLabel(Dialog)
        self.eps_label.setObjectName("eps_label")
        self.gridLayout_2.addWidget(self.eps_label, 11, 1, 1, 1)
        self.xp_label = QtWidgets.QLabel(Dialog)
        self.xp_label.setObjectName("xp_label")
        self.gridLayout_2.addWidget(self.xp_label, 10, 1, 1, 1)
        self.analysis_progressbar = QtWidgets.QProgressBar(Dialog)
        self.analysis_progressbar.setProperty("value", 0)
        self.analysis_progressbar.setObjectName("analysis_progressbar")
        self.gridLayout_2.addWidget(self.analysis_progressbar, 8, 0, 1, 2)
        self.label_11 = QtWidgets.QLabel(Dialog)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 1, 0, 1, 1)
        self.label_28 = QtWidgets.QLabel(Dialog)
        self.label_28.setObjectName("label_28")
        self.gridLayout_2.addWidget(self.label_28, 12, 0, 1, 1)
        self.label_20 = QtWidgets.QLabel(Dialog)
        self.label_20.setObjectName("label_20")
        self.gridLayout_2.addWidget(self.label_20, 10, 0, 1, 1)
        self.image_progressbar = QtWidgets.QProgressBar(Dialog)
        self.image_progressbar.setProperty("value", 0)
        self.image_progressbar.setObjectName("image_progressbar")
        self.gridLayout_2.addWidget(self.image_progressbar, 0, 0, 1, 2)
        self.horizontalLayout.addLayout(self.gridLayout_2)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_16.setText(_translate("Dialog", "Dataset:"))
        self.dataset_select_label.setText(_translate("Dialog", "Dataset: -- positions x -- images"))
        self.label_3.setText(_translate("Dialog", "ROI height"))
        self.label_18.setText(_translate("Dialog", "Rotation / deg"))
        self.label_10.setText(_translate("Dialog", "Bkg cut"))
        self.label_4.setText(_translate("Dialog", "ROI left"))
        self.label_14.setText(_translate("Dialog", "Mask kernel"))
        self.label_9.setText(_translate("Dialog", "Show"))
        self.label_5.setText(_translate("Dialog", "ROI width"))
        self.raw_radiobutton.setText(_translate("Dialog", "Raw"))
        self.processed_radiobutton.setText(_translate("Dialog", "Proc"))
        self.tight_radiobutton.setText(_translate("Dialog", "Tight"))
        self.image_size_label.setText(_translate("Dialog", "-- x --"))
        self.label_23.setText(_translate("Dialog", "Beam energy / MeV"))
        self.label_22.setText(_translate("Dialog", "Slit - screen distance / cm"))
        self.label_8.setText(_translate("Dialog", "Image size"))
        self.label_2.setText(_translate("Dialog", "ROI top"))
        self.label.setText(_translate("Dialog", "Med filt kernel"))
        self.label_21.setText(_translate("Dialog", "Pixel size / um"))
        self.label_26.setText(_translate("Dialog", "Analyze scan"))
        self.start_analysis_button.setText(_translate("Dialog", "Start"))
        self.label_25.setText(_translate("Dialog", "Rejected"))
        self.charge_label.setText(_translate("Dialog", "---"))
        self.moment_1_label.setText(_translate("Dialog", "-.--"))
        self.eps_n_label.setText(_translate("Dialog", "-.--"))
        self.label_6.setText(_translate("Dialog", "xp"))
        self.label_15.setText(_translate("Dialog", "Charge"))
        self.label_19.setText(_translate("Dialog", "Beam width sigma"))
        self.label_24.setText(_translate("Dialog", "Emittance"))
        self.label_7.setText(_translate("Dialog", "xp^2"))
        self.bkg_label.setText(_translate("Dialog", "-.-"))
        self.rejected_images_label.setText(_translate("Dialog", "-"))
        self.moment_2_label.setText(_translate("Dialog", "-.--"))
        self.sigma_x_label.setText(_translate("Dialog", "-.--"))
        self.eps_label.setText(_translate("Dialog", "-.--"))
        self.xp_label.setText(_translate("Dialog", "-.--"))
        self.label_11.setText(_translate("Dialog", "Bkg level"))
        self.label_28.setText(_translate("Dialog", "Normalized emittance"))
        self.label_20.setText(_translate("Dialog", "Beam divergence"))
from pyqtgraph import ImageView, PlotWidget
