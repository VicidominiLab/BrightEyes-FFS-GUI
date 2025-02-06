from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import QSize
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib import gridspec

class CustomNavigationToolbar(NavigationToolbar2QT):
    # Define the buttons you want to show
    toolitems = [
        ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        ('Save', 'Save the figure', 'filesave', 'save_figure'),
    ]
    def set_message(self, s):
        # Override the set_message method to disable displaying cursor coordinates
        pass
    
    
class MplWidgetCorrPlot(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        
        my_dpi = 120
        h = Figure(figsize=(350/my_dpi, 350/my_dpi), dpi=my_dpi)
        h.set_facecolor('#FAFAFA')
        
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        
        self.canvas = FigureCanvas(h)
        self.toolbar = CustomNavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #FAFAFA;")
        icon_size = QSize(12, 12)
        self.toolbar.setIconSize(icon_size)
        
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        vertical_layout.addWidget(self.toolbar)
        
        
        
        self.canvas.axes = self.canvas.figure.add_subplot(gs[0])
        #self.canvas.axes.plot([1, 2, 3], [4, 5, 6])  # Example data for subplot 1
        
        self.canvas.axes2 = self.canvas.figure.add_subplot(gs[1])
        #self.canvas.axes2.plot([1, 2, 3], [1, 2, 3])  # Example data for subplot 2
        
        self.canvas.figure.subplots_adjust(bottom=0.15, top=0.95, left=0.2, right=0.95, hspace=0.4)
        
        self.setLayout(vertical_layout)
