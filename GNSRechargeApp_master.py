import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget
from PyQt5 import QtGui

class MainWindow(QMainWindow):                                                                                      #funtional animal with animal in it

    def __init__(self):                                                                                             # setting up organs of animal giving Mainwindow animal
        super().__init__()                                                                                          # super() lets the Mainwindow animal inherit organs from Qmain, functions of Qmain liturally cary over
        self.title = 'Recharge forcasting App'#'PyQt5 tabs - pythonspot.com'
        self.left = 0
        self.top = 0
        self.width = 300
        self.height = 200
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.icon = QtGui.QIcon('gns.ico')
        self.setWindowIcon(self.icon)
        self.setStyleSheet("background-color: #a41c2c;")

        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)

        self.show()


class MyTableWidget(QWidget):

    tab3: QWidget

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabBar {background: #f0f0f0; color: #404040; font: bold large New Century Schoolbook; font-size: 12pt; font-style: bold}; QTabBar::tab {border-width: 0px;}") #ff000000;font-family: OldEnglish;font-size: 12pt;}")
        #self.tabs.setStyleSheet("QTabBar::tab {background-color: black;")
        #self.tabs.setStyleSheet('setStyleSheet("QTabWidget::pane {border-bottom: 30px;}")')
        self.tab1 = QWidget()
        self.tab1.setStyleSheet("background-color: #fafafa")
        self.tab2 = QWidget()
        self.tab2.setStyleSheet("background-color: #fafafa")
        self.tab3 = QWidget()
        self.tab3.setStyleSheet("background-color: #fafafa")
        self.tabs.resize(300, 200)


        # Add tabs
        self.tabs.addTab(self.tab1, "NGRM - Forecaster")
        self.tabs.addTab(self.tab2, "NGRM - Uncertainty")
        self.tabs.addTab(self.tab3, "NGRM - Validation")
        # self.tabs.addTab(self.tab1, "MODEL2 - Forecaster")
        # self.tabs.addTab(self.tab2, "MODEL2 - Uncertainty")
        # self.tabs.addTab(self.tab3, "MODEL2 - Validation")


        # Create first tab
        # self.tab1.layout = QVBoxLayout(self)
        # self.pushButton1 = QPushButton("PyQt5 button")
        # self.tab1.layout.addWidget(self.pushButton1)
        # self.tab1.setLayout(self.tab1.layout)

        web = QWebEngineView()                                                          #create web objetect (not a property of tab object)
        web.load(QUrl("https://hijisvanadel.users.earthengine.app/view/gnsrecharge1"))
        web2 = QWebEngineView()                                                          #create web objetect (not a property of tab object)
        web2.load(QUrl("https://hijisvanadel.users.earthengine.app/view/gnsapp"))

        self.tab1.layout = QGridLayout(self)                                           # create new object layout, and use Qgrid ond the tab1 widget (self)
        self.tab1.layout.addWidget(web, 0, 0, 2, 1)                                    # to the tab1 gridlayout object tabs, add a widget to the grid
        self.tab1.setLayout(self.tab1.layout)                                          # self.tab1.layout   being the Qrigdlayout object  should be the layout

        self.tab2.layout = QGridLayout(self)                                           # create new object layout, and use Qgrid ond the tab1 widget (self)
        self.tab2.layout.addWidget(web2, 0, 0, 2, 1)                                    # to the tab1 gridlayout object tabs, add a widget to the grid
        self.tab2.setLayout(self.tab2.layout)                                          # self.tab1.layout   being the Qrigdlayout object  should be the layout

        #image?


        #self.label.setPixmap(QtGui.QPixmap("AllR.png"))
        #self.vbox.addWidget(self.label)
        # vbox.addWidget(self.tab3)
        # self.tab3.setLayout(vbox)

        # self.label = QLabel(self)
        # self.pixmap = QPixmap('AllR.png')
        # self.label.setPixmap(self.pixmap)

        # self.tab3.layout = QGridLayout(self)                                           # create new object layout, and use Qgrid ond the tab1 widget (self)
        # self.pushButton1 = QPushButton("PyQt5 button")
        # self.tab3.layout.addWidget(self.pushButton1)                                    # to the tab1 gridlayout object tabs, add a widget to the grid
        # self.tab3.setLayout(self.tab3.layout)                                          # self.tab1.layout   being the Qrigdlayout object  should be the layout

        # self.tab3.label = QLabel("Yellowcccc", self) #QtGui.QLabel(self.tab3)
        # self.pixmap = QPixmap('AllR.png').scaled(2020, 405,Qt.KeepAspectRatio)
        # self.tab3.label.setPixmap(self.pixmap)
        # self.tab3.layout = QGridLayout(self)
        # self.tab3.layout.addWidget(self.tab3.label)
        # self.tab3.setLayout(self.tab3.layout)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)



    @pyqtSlot()
    def on_click(self):
        print("\n")
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())


if __name__ == '__main__':                                                                 # if terminal is running current script (not imported)
    app = QApplication(sys.argv)                                                           # initialise C++ pyqt5 app
    w1 = MainWindow()                                                                      # define instant/object of the self created MainWindow class containing subclass Qmainwindow,  Qmainwindow is an existing class from which our newly defined class 'Mainwindow' inherits functionality through super()
    w1.showMaximized()


    #w1.show()
    #layout grid



    sys.exit(app.exec_())