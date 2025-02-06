from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

    
class MplWidgetDiffLawPlot(QWidget):
    
    def __init__(self, parent = None):

        QWidget.__init__(self, parent)
        
        my_dpi = 120
        h = Figure(figsize=(350/my_dpi, 350/my_dpi), dpi=my_dpi)
        h.set_facecolor('#FAFAFA')
        
        self.canvas = FigureCanvas(h)
        
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.canvas.figure.subplots_adjust(bottom=0.2, top=0.95, left=0.15, right=0.95)
        self.setLayout(vertical_layout)
