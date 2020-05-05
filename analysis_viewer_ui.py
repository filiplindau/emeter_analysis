# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\analysis_viewer_ui.ui'
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
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setContentsMargins(-1, -1, -1, 6)
        self.gridLayout.setObjectName("gridLayout")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 14, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 15, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(Dialog)
        self.label_12.setText("")
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 5, 0, 1, 1)
        self.moment_2_label = QtWidgets.QLabel(Dialog)
        self.moment_2_label.setObjectName("moment_2_label")
        self.gridLayout.addWidget(self.moment_2_label, 15, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(Dialog)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 13, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.roi_height_spinbox = QtWidgets.QSpinBox(Dialog)
        self.roi_height_spinbox.setMaximum(3000)
        self.roi_height_spinbox.setProperty("value", 700)
        self.roi_height_spinbox.setObjectName("roi_height_spinbox")
        self.gridLayout.addWidget(self.roi_height_spinbox, 2, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.roi_left_spinbox = QtWidgets.QSpinBox(Dialog)
        self.roi_left_spinbox.setMaximum(3000)
        self.roi_left_spinbox.setObjectName("roi_left_spinbox")
        self.gridLayout.addWidget(self.roi_left_spinbox, 3, 1, 1, 1)
        self.bkg_spinbox = QtWidgets.QSpinBox(Dialog)
        self.bkg_spinbox.setMaximumSize(QtCore.QSize(16777209, 16777215))
        self.bkg_spinbox.setMaximum(65000)
        self.bkg_spinbox.setProperty("value", 10)
        self.bkg_spinbox.setObjectName("bkg_spinbox")
        self.gridLayout.addWidget(self.bkg_spinbox, 7, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.bkg_label = QtWidgets.QLabel(Dialog)
        self.bkg_label.setObjectName("bkg_label")
        self.gridLayout.addWidget(self.bkg_label, 13, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 4, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(Dialog)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 7, 0, 1, 1)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 6, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(Dialog)
        self.label_13.setText("")
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 8, 0, 1, 1)
        self.roi_width_spinbox = QtWidgets.QSpinBox(Dialog)
        self.roi_width_spinbox.setMaximum(3000)
        self.roi_width_spinbox.setProperty("value", 1400)
        self.roi_width_spinbox.setObjectName("roi_width_spinbox")
        self.gridLayout.addWidget(self.roi_width_spinbox, 4, 1, 1, 1)
        self.label_9 = QtWidgets.QLabel(Dialog)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 10, 0, 1, 1)
        self.medfilt_spinbox = QtWidgets.QSpinBox(Dialog)
        self.medfilt_spinbox.setMinimum(1)
        self.medfilt_spinbox.setMaximum(13)
        self.medfilt_spinbox.setProperty("value", 5)
        self.medfilt_spinbox.setObjectName("medfilt_spinbox")
        self.gridLayout.addWidget(self.medfilt_spinbox, 6, 1, 1, 1)
        self.roi_top_spinbox = QtWidgets.QSpinBox(Dialog)
        self.roi_top_spinbox.setMaximum(3000)
        self.roi_top_spinbox.setProperty("value", 250)
        self.roi_top_spinbox.setObjectName("roi_top_spinbox")
        self.gridLayout.addWidget(self.roi_top_spinbox, 1, 1, 1, 1)
        self.moment_1_label = QtWidgets.QLabel(Dialog)
        self.moment_1_label.setObjectName("moment_1_label")
        self.gridLayout.addWidget(self.moment_1_label, 14, 1, 1, 1)
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
        self.gridLayout.addLayout(self.horizontalLayout_2, 10, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem, 11, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(Dialog)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 9, 0, 1, 1)
        self.image_size_label = QtWidgets.QLabel(Dialog)
        self.image_size_label.setObjectName("image_size_label")
        self.gridLayout.addWidget(self.image_size_label, 9, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.file_listview = QtWidgets.QListView(Dialog)
        self.file_listview.setObjectName("file_listview")
        self.verticalLayout.addWidget(self.file_listview)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.image_widget = ImageView(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image_widget.sizePolicy().hasHeightForWidth())
        self.image_widget.setSizePolicy(sizePolicy)
        self.image_widget.setObjectName("image_widget")
        self.horizontalLayout.addWidget(self.image_widget)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_6.setText(_translate("Dialog", "1st moment"))
        self.label_7.setText(_translate("Dialog", "2nd moment"))
        self.moment_2_label.setText(_translate("Dialog", "-.--"))
        self.label_11.setText(_translate("Dialog", "Bkg level"))
        self.label_2.setText(_translate("Dialog", "ROI top"))
        self.label_4.setText(_translate("Dialog", "ROI left"))
        self.label_3.setText(_translate("Dialog", "ROI height"))
        self.bkg_label.setText(_translate("Dialog", "-.-"))
        self.label_5.setText(_translate("Dialog", "ROI width"))
        self.label_10.setText(_translate("Dialog", "Bkg cut"))
        self.label.setText(_translate("Dialog", "Med filt kernel"))
        self.label_9.setText(_translate("Dialog", "Show"))
        self.moment_1_label.setText(_translate("Dialog", "-.--"))
        self.raw_radiobutton.setText(_translate("Dialog", "Raw"))
        self.processed_radiobutton.setText(_translate("Dialog", "Proc"))
        self.tight_radiobutton.setText(_translate("Dialog", "Tight"))
        self.label_8.setText(_translate("Dialog", "Image size"))
        self.image_size_label.setText(_translate("Dialog", "-- x --"))
from pyqtgraph import ImageView