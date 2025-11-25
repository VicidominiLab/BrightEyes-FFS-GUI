from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMenu, QAction
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

    
class MplWidgetLinePlot(QWidget):
    copyBadChunks = pyqtSignal()
    turnOffBadChunks = pyqtSignal()
    turnOnAllChunks = pyqtSignal()
    filterBadChunks = pyqtSignal()
    
    def __init__(self, parent = None):

        QWidget.__init__(self, parent)
        
        my_dpi = 120
        h = Figure(figsize=(350/my_dpi, 350/my_dpi), dpi=my_dpi)
        h.set_facecolor('#FAFAFA')
        
        self.canvas = FigureCanvas(h)
        
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.canvas.figure.subplots_adjust(bottom=0.3, top=0.95, right=0.95)
        self.setLayout(vertical_layout)
        
        
        # Enable mouse events
        self.setContextMenuPolicy(0)  # Default behavior so we handle right-click manually
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)

    def on_mouse_click(self, event):
        if event.button == 3:  # Right-click
            self.show_context_menu(event)

    def show_context_menu(self, event):
        menu = QMenu(self)

        action0 = QAction("Filter out bad chunks (beta)", self)
        action0.triggered.connect(lambda: self.filterBadChunks.emit())
        menu.addAction(action0)

        action1 = QAction("Copy list of bad chunks", self)
        action1.triggered.connect(lambda: self.copyBadChunks.emit())
        menu.addAction(action1)
    
        action2 = QAction("Turn off bad chunks from list", self)
        action2.triggered.connect(lambda: self.turnOffBadChunks.emit())
        menu.addAction(action2)
        
        action3 = QAction("Turn on all chunks", self)
        action3.triggered.connect(lambda: self.turnOnAllChunks.emit())
        menu.addAction(action3)
    
        # Use QCursor to get global mouse position
        menu.exec_(QCursor.pos())
