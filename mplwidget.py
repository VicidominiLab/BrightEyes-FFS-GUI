from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

    
class MplWidget(QWidget):
    
    def __init__(self, parent = None):

        QWidget.__init__(self, parent)
        
        my_dpi = 120
        h = Figure(figsize=(110/my_dpi, 110/my_dpi), dpi=my_dpi)
        h.set_facecolor('#FAFAFA')
        
        self.canvas = FigureCanvas(h)
        
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.canvas.figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        self.setLayout(vertical_layout)
