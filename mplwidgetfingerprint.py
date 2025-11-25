from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMenu, QAction
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

    
class MplWidgetFingerPrint(QWidget):
    showPhotons = pyqtSignal()
    showN = pyqtSignal()
    showTau = pyqtSignal()
    showD = pyqtSignal()
    showw0 = pyqtSignal()
    
    def __init__(self, parent = None):

        QWidget.__init__(self, parent)
        
        my_dpi = 120
        h = Figure(figsize=(110/my_dpi, 110/my_dpi), dpi=my_dpi)
        h.set_facecolor('#FAFAFA')
        
        self.canvas = FigureCanvas(h)
        
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.canvas.figure.subplots_adjust(left=0, bottom=0, right=0.6, top=1, wspace=None, hspace=None)
        self.setLayout(vertical_layout)

        # Enable mouse events
        self.setContextMenuPolicy(0)  # Default behavior so we handle right-click manually
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)

    def on_mouse_click(self, event):
        if event.button == 3:  # Right-click
            self.show_context_menu(event)
    
    def show_context_menu(self, event):
        menu = QMenu(self)
    
        action1 = QAction("Show total number of photons", self)
        action1.triggered.connect(lambda: self.showPhotons.emit())
        menu.addAction(action1)
    
        action2 = QAction("Show N", self)
        action2.triggered.connect(lambda: self.showN.emit())
        menu.addAction(action2)
        
        action3 = QAction("Show tau", self)
        action3.triggered.connect(lambda: self.showTau.emit())
        menu.addAction(action3)
        
        action3b = QAction("Show D", self)
        action3b.triggered.connect(lambda: self.showD.emit())
        menu.addAction(action3b)
        
        action4 = QAction("Show beam waist", self)
        action4.triggered.connect(lambda: self.showw0.emit())
        menu.addAction(action4)
    
        # Use QCursor to get global mouse position
        menu.exec_(QCursor.pos())