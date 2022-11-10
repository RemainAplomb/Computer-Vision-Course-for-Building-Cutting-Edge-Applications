# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'name.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.labelbutton = QtWidgets.QPushButton(self.centralwidget, clicked= lambda : self.btn_clicked())
        self.labelbutton.setGeometry(QtCore.QRect(150, 30, 441, 221))
        font = QtGui.QFont()
        font.setFamily("Perpetua")
        font.setPointSize(28)
        self.labelbutton.setFont(font)
        self.labelbutton.setObjectName("labelbutton")
        self.namelabel = QtWidgets.QLabel(self.centralwidget)
        self.namelabel.setGeometry(QtCore.QRect(270, 350, 481, 161))
        font = QtGui.QFont()
        font.setFamily("Modern No. 20")
        font.setPointSize(28)
        self.namelabel.setFont(font)
        self.namelabel.setObjectName("namelabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.menuMyname = QtWidgets.QMenu(self.menubar)
        self.menuMyname.setObjectName("menuMyname")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuMyname.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def btn_clicked(self):
        self.namelabel.setText("My name is James Bond")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.labelbutton.setText(_translate("MainWindow", " Clickable Button"))
        self.namelabel.setText(_translate("MainWindow", "My Name is: "))
        self.menuMyname.setTitle(_translate("MainWindow", "Myname"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
