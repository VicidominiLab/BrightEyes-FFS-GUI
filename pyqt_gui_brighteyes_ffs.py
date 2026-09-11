# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'brighteyes_ffs_3.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# Modernized DPI-aware presentation layer based on the generated UI file (v14).
# Widget names, actions, and signal wiring are intentionally preserved.
# Regenerating this file with pyuic5 will overwrite the custom theme below.


from PyQt5 import QtCore, QtGui, QtWidgets

# Make Qt use the same logical-size model when a laptop is connected
# to an external monitor. These attributes only take effect if this
# module is imported before QApplication is created; otherwise they are
# harmless.
try:
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    if hasattr(QtCore.Qt, "HighDpiScaleFactorRoundingPolicy"):
        QtCore.QCoreApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
except Exception:
    pass


try:
    import matplotlib as _mpl
except Exception:
    _mpl = None


def _clamp(value, low, high):
    return max(low, min(high, value))


def _screen_ui_scale(widget=None):
    """Return a conservative UI scale for high-DPI/small laptop screens.

    The default desktop look remains unchanged at 1.0. On high-DPI screens,
    sizes are increased only modestly so the GUI does not become oversized on
    a normal external monitor.
    """
    app = QtWidgets.QApplication.instance()
    screen = None
    if widget is not None and hasattr(widget, 'screen'):
        screen = widget.screen()
    if screen is None and app is not None:
        screen = app.primaryScreen()
    if screen is None:
        return 1.0

    dpi = screen.logicalDotsPerInch() or 96.0
    scale = dpi / 96.0

    # Keep scaling modest. This fixes tiny laptop text without making the
    # already-good desktop layout look huge.
    scale = _clamp(scale, 1.0, 1.14)

    # Very small logical screens need layout space more than large widgets.
    height = screen.availableGeometry().height()
    if height < 850:
        scale = min(scale, 1.06)
    elif height < 950:
        scale = min(scale, 1.10)

    return scale


def _scaled(value, scale):
    return int(round(value * scale))


def _make_arrow_icon(direction, color="#334155"):
    """Create symmetric navigation-arrow icons.

    Qt/platform standard arrows can have different intrinsic padding on
    different systems. Drawing both icons from the same geometry keeps the
    left and right buttons visually identical.
    """
    pixmap = QtGui.QPixmap(24, 24)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtGui.QColor(color))

    # Slightly larger, darker arrows: still symmetric, but easier to see
    # on both light-gray navigation buttons and high-DPI displays.
    if direction == "left":
        points = [QtCore.QPointF(6, 12), QtCore.QPointF(17, 4), QtCore.QPointF(17, 20)]
    else:
        points = [QtCore.QPointF(18, 12), QtCore.QPointF(7, 4), QtCore.QPointF(7, 20)]

    painter.drawPolygon(QtGui.QPolygonF(points))
    painter.end()
    return QtGui.QIcon(pixmap)


class _MenuPolisher(QtCore.QObject):
    """Keep application and context menus visually consistent.

    Some platform styles draw standard QAction icons on dark square tiles.
    The menus remain fully functional; only those menu icons are removed.
    """

    def eventFilter(self, obj, event):
        if isinstance(obj, QtWidgets.QMenu) and event.type() == QtCore.QEvent.Show:
            for action in obj.actions():
                if not action.icon().isNull():
                    action.setIcon(QtGui.QIcon())
        return super().eventFilter(obj, event)


class _CleanComboItemDelegate(QtWidgets.QStyledItemDelegate):
    """Draw combo-box entries without native marker placeholders."""

    def __init__(self, parent=None, minimum_height=21):
        super().__init__(parent)
        self.minimum_height = minimum_height

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.features &= ~QtWidgets.QStyleOptionViewItem.HasCheckIndicator
        option.checkState = QtCore.Qt.Unchecked
        option.icon = QtGui.QIcon()
        option.decorationSize = QtCore.QSize(0, 0)

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(max(self.minimum_height, option.fontMetrics.height() + 4))
        return size


class _FileTabButton(QtWidgets.QPushButton):
    """File selector button with deterministic custom painting.

    Drawing the background here prevents platform or application styles from
    replacing the requested colors with the native grey button appearance.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setAttribute(QtCore.Qt.WA_Hover, True)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        hovered = self.underMouse()
        pressed = self.isDown()
        active = self.isChecked()

        if active:
            background = '#4f46e5'
            border = '#4f46e5'
            foreground = '#ffffff'
            if pressed:
                background = border = '#3730a3'
            elif hovered:
                background = border = '#4338ca'
        else:
            background = '#eef2ff'
            border = '#c7d2fe'
            foreground = '#3730a3'
            if pressed:
                background = '#cfd7ff'
                border = '#4f46e5'
            elif hovered:
                background = '#dfe5ff'
                border = '#6366f1'
                foreground = '#312e81'

        # Keep disabled file slots visibly part of the same file-tab family
        # instead of reverting to the platform's grey native-button color.
        if not self.isEnabled() and not active:
            background = '#f1f3ff'
            border = '#d8dcf6'
            foreground = '#858bb6'

        painter.setPen(QtGui.QPen(QtGui.QColor(border), 1.0))
        painter.setBrush(QtGui.QColor(background))
        painter.drawRoundedRect(rect, 5.0, 5.0)

        painter.setPen(QtGui.QColor(foreground))
        painter.setFont(self.font())
        text_rect = self.rect().adjusted(7, 0, -7, 0)
        painter.drawText(
            text_rect,
            QtCore.Qt.AlignCenter | QtCore.Qt.TextSingleLine,
            self.text(),
        )


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1272, 937)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMinimumSize(QtCore.QSize(500, 0))
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.central_window = QtWidgets.QVBoxLayout()
        self.central_window.setObjectName("central_window")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.prevFCSfile_button = QtWidgets.QPushButton(self.centralwidget)
        self.prevFCSfile_button.setObjectName("prevFCSfile_button")
        self.horizontalLayout_3.addWidget(self.prevFCSfile_button)
        self.FCSfile0_button = _FileTabButton(self.centralwidget)
        self.FCSfile0_button.setObjectName("FCSfile0_button")
        self.horizontalLayout_3.addWidget(self.FCSfile0_button)
        self.FCSfile1_button = _FileTabButton(self.centralwidget)
        self.FCSfile1_button.setObjectName("FCSfile1_button")
        self.horizontalLayout_3.addWidget(self.FCSfile1_button)
        self.FCSfile2_button = _FileTabButton(self.centralwidget)
        self.FCSfile2_button.setObjectName("FCSfile2_button")
        self.horizontalLayout_3.addWidget(self.FCSfile2_button)
        self.FCSfile3_button = _FileTabButton(self.centralwidget)
        self.FCSfile3_button.setObjectName("FCSfile3_button")
        self.horizontalLayout_3.addWidget(self.FCSfile3_button)
        self.FCSfile4_button = _FileTabButton(self.centralwidget)
        self.FCSfile4_button.setObjectName("FCSfile4_button")
        self.horizontalLayout_3.addWidget(self.FCSfile4_button)
        self.nextFCSfile_button = QtWidgets.QPushButton(self.centralwidget)
        self.nextFCSfile_button.setIconSize(QtCore.QSize(20, 20))
        self.nextFCSfile_button.setObjectName("nextFCSfile_button")
        self.horizontalLayout_3.addWidget(self.nextFCSfile_button)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 3)
        self.horizontalLayout_3.setStretch(2, 3)
        self.horizontalLayout_3.setStretch(3, 3)
        self.horizontalLayout_3.setStretch(4, 3)
        self.horizontalLayout_3.setStretch(5, 3)
        self.horizontalLayout_3.setStretch(6, 1)
        self.central_window.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.FCSFileName_label = QtWidgets.QLabel(self.centralwidget)
        self.FCSFileName_label.setObjectName("FCSFileName_label")
        self.gridLayout_5.addWidget(self.FCSFileName_label, 1, 0, 1, 1)
        self.FCSFolderName_label = QtWidgets.QLabel(self.centralwidget)
        self.FCSFolderName_label.setObjectName("FCSFolderName_label")
        self.gridLayout_5.addWidget(self.FCSFolderName_label, 0, 0, 1, 1)
        self.horizontalLayout_4.addLayout(self.gridLayout_5)
        self.label_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.label_edit.setObjectName("label_edit")
        self.horizontalLayout_4.addWidget(self.label_edit)
        self.ycoord_edit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.ycoord_edit.setFont(font)
        self.ycoord_edit.setObjectName("ycoord_edit")
        self.horizontalLayout_4.addWidget(self.ycoord_edit)
        self.xcoord_edit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.xcoord_edit.setFont(font)
        self.xcoord_edit.setObjectName("xcoord_edit")
        self.horizontalLayout_4.addWidget(self.xcoord_edit)
        self.saveLabel_button = QtWidgets.QPushButton(self.centralwidget)
        self.saveLabel_button.setObjectName("saveLabel_button")
        self.horizontalLayout_4.addWidget(self.saveLabel_button)
        self.horizontalLayout_4.setStretch(0, 10)
        self.horizontalLayout_4.setStretch(1, 3)
        self.horizontalLayout_4.setStretch(2, 1)
        self.horizontalLayout_4.setStretch(3, 1)
        self.horizontalLayout_4.setStretch(4, 1)
        self.central_window.addLayout(self.horizontalLayout_4)
        spacerItem = QtWidgets.QSpacerItem(40, 8, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.central_window.addItem(spacerItem)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.central_window.addLayout(self.verticalLayout_5)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.correlations_treeWidget = QtWidgets.QTreeWidget(self.centralwidget)
        self.correlations_treeWidget.setObjectName("correlations_treeWidget")
        self.horizontalLayout_5.addWidget(self.correlations_treeWidget)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.fingerprint_widget = MplWidgetFingerPrint(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fingerprint_widget.sizePolicy().hasHeightForWidth())
        self.fingerprint_widget.setSizePolicy(sizePolicy)
        self.fingerprint_widget.setMaximumSize(QtCore.QSize(150, 150))
        self.fingerprint_widget.setObjectName("fingerprint_widget")
        self.horizontalLayout_5.addWidget(self.fingerprint_widget)
        self.horizontalLayout_5.setStretch(0, 40)
        self.horizontalLayout_5.setStretch(2, 10)
        self.central_window.addLayout(self.horizontalLayout_5)
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.timetrace_widget = MplWidgetLinePlot(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.timetrace_widget.sizePolicy().hasHeightForWidth())
        self.timetrace_widget.setSizePolicy(sizePolicy)
        self.timetrace_widget.setBaseSize(QtCore.QSize(10, 10))
        self.timetrace_widget.setObjectName("timetrace_widget")
        self.gridLayout_6.addWidget(self.timetrace_widget, 0, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.showElements_widget = QtWidgets.QListWidget(self.centralwidget)
        self.showElements_widget.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.showElements_widget.sizePolicy().hasHeightForWidth())
        self.showElements_widget.setSizePolicy(sizePolicy)
        self.showElements_widget.setMinimumSize(QtCore.QSize(120, 58))
        self.showElements_widget.setMaximumSize(QtCore.QSize(120, 58))
        self.showElements_widget.setObjectName("showElements_widget")
        item = QtWidgets.QListWidgetItem()
        self.showElements_widget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.showElements_widget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.showElements_widget.addItem(item)
        self.verticalLayout_2.addWidget(self.showElements_widget)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.chunk_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.chunk_spinBox.setMaximum(10000)
        self.chunk_spinBox.setObjectName("chunk_spinBox")
        self.horizontalLayout_6.addWidget(self.chunk_spinBox)
        self.chunkOn_checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.chunkOn_checkBox.setShortcut("")
        self.chunkOn_checkBox.setObjectName("chunkOn_checkBox")
        self.horizontalLayout_6.addWidget(self.chunkOn_checkBox)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        self.verticalLayout_2.setStretch(0, 10)
        self.verticalLayout_2.setStretch(1, 2)
        self.gridLayout_6.addLayout(self.verticalLayout_2, 0, 1, 1, 1)
        self.gridLayout_6.setColumnStretch(0, 7)
        self.gridLayout_6.setColumnStretch(1, 4)
        self.central_window.addLayout(self.gridLayout_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.showchunkscorr_dropdown = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.showchunkscorr_dropdown.sizePolicy().hasHeightForWidth())
        self.showchunkscorr_dropdown.setSizePolicy(sizePolicy)
        self.showchunkscorr_dropdown.setMinimumSize(QtCore.QSize(140, 0))
        self.showchunkscorr_dropdown.setMaximumSize(QtCore.QSize(250, 16777215))
        self.showchunkscorr_dropdown.setObjectName("showchunkscorr_dropdown")
        self.showchunkscorr_dropdown.addItem("")
        self.showchunkscorr_dropdown.addItem("")
        self.verticalLayout_3.addWidget(self.showchunkscorr_dropdown)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.correlations_widget = MplWidgetCorrPlot(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.correlations_widget.sizePolicy().hasHeightForWidth())
        self.correlations_widget.setSizePolicy(sizePolicy)
        self.correlations_widget.setAutoFillBackground(False)
        self.correlations_widget.setObjectName("correlations_widget")
        self.horizontalLayout_8.addWidget(self.correlations_widget)
        self.verticalLayout_3.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_7.addLayout(self.verticalLayout_3)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(4, -1, 6, -1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.difflaw_dropdown = QtWidgets.QComboBox(self.centralwidget)
        self.difflaw_dropdown.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.difflaw_dropdown.sizePolicy().hasHeightForWidth())
        self.difflaw_dropdown.setSizePolicy(sizePolicy)
        self.difflaw_dropdown.setMinimumSize(QtCore.QSize(120, 0))
        self.difflaw_dropdown.setObjectName("difflaw_dropdown")
        self.difflaw_dropdown.addItem("")
        self.difflaw_dropdown.addItem("")
        self.difflaw_dropdown.addItem("")
        self.verticalLayout.addWidget(self.difflaw_dropdown)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.difflaw_widget = MplWidgetDiffLawPlot(self.centralwidget)
        self.difflaw_widget.setObjectName("difflaw_widget")
        self.horizontalLayout_10.addWidget(self.difflaw_widget)
        self.verticalLayout.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_7.addLayout(self.verticalLayout)
        self.central_window.addLayout(self.horizontalLayout_7)
        self.central_window.setStretch(0, 1)
        self.central_window.setStretch(4, 3)
        self.central_window.setStretch(5, 3)
        self.central_window.setStretch(6, 10)
        self.gridLayout_8.addLayout(self.central_window, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1272, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuCalculate = QtWidgets.QMenu(self.menubar)
        self.menuCalculate.setObjectName("menuCalculate")
        self.menuRemove = QtWidgets.QMenu(self.menubar)
        self.menuRemove.setObjectName("menuRemove")
        self.menuTools = QtWidgets.QMenu(self.menubar)
        self.menuTools.setObjectName("menuTools")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.dockWidget = QtWidgets.QDockWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dockWidget.sizePolicy().hasHeightForWidth())
        self.dockWidget.setSizePolicy(sizePolicy)
        self.dockWidget.setMinimumSize(QtCore.QSize(312, 145))
        font = QtGui.QFont()
        font.setStrikeOut(False)
        self.dockWidget.setFont(font)
        self.dockWidget.setAutoFillBackground(False)
        self.dockWidget.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable|QtWidgets.QDockWidget.DockWidgetMovable)
        self.dockWidget.setObjectName("dockWidget")
        self.dockWidgetContents = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dockWidgetContents.sizePolicy().hasHeightForWidth())
        self.dockWidgetContents.setSizePolicy(sizePolicy)
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.dockWidgetContents)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.frame = QtWidgets.QFrame(self.dockWidgetContents)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.image_widget = MplWidget(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image_widget.sizePolicy().hasHeightForWidth())
        self.image_widget.setSizePolicy(sizePolicy)
        self.image_widget.setObjectName("image_widget")
        self.gridLayout_12.addWidget(self.image_widget, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.frame, 0, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_2, 0, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.prevImage_button = QtWidgets.QPushButton(self.dockWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.prevImage_button.sizePolicy().hasHeightForWidth())
        self.prevImage_button.setSizePolicy(sizePolicy)
        self.prevImage_button.setObjectName("prevImage_button")
        self.horizontalLayout.addWidget(self.prevImage_button)
        self.imageName_button = QtWidgets.QPushButton(self.dockWidgetContents)
        self.imageName_button.setObjectName("imageName_button")
        self.horizontalLayout.addWidget(self.imageName_button)
        self.nextImage_button = QtWidgets.QPushButton(self.dockWidgetContents)
        self.nextImage_button.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.nextImage_button.sizePolicy().hasHeightForWidth())
        self.nextImage_button.setSizePolicy(sizePolicy)
        self.nextImage_button.setIconSize(QtCore.QSize(16, 16))
        self.nextImage_button.setObjectName("nextImage_button")
        self.horizontalLayout.addWidget(self.nextImage_button)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 10)
        self.horizontalLayout.setStretch(2, 1)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.imageInfo_label = QtWidgets.QLabel(self.dockWidgetContents)
        self.imageInfo_label.setObjectName("imageInfo_label")
        self.gridLayout_3.addWidget(self.imageInfo_label, 0, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_3, 2, 0, 1, 1)
        self.gridLayout.setRowStretch(0, 100)
        self.gridLayout.setRowStretch(1, 10)
        self.gridLayout.setRowStretch(2, 30)
        self.gridLayout_11.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.dockWidget.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget)
        self.dockWidget_6 = QtWidgets.QDockWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dockWidget_6.sizePolicy().hasHeightForWidth())
        self.dockWidget_6.setSizePolicy(sizePolicy)
        self.dockWidget_6.setMinimumSize(QtCore.QSize(160, 160))
        self.dockWidget_6.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable|QtWidgets.QDockWidget.DockWidgetMovable)
        self.dockWidget_6.setObjectName("dockWidget_6")
        self.dockWidgetContents_6 = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dockWidgetContents_6.sizePolicy().hasHeightForWidth())
        self.dockWidgetContents_6.setSizePolicy(sizePolicy)
        self.dockWidgetContents_6.setObjectName("dockWidgetContents_6")
        self.notes_edit = QtWidgets.QPlainTextEdit(self.dockWidgetContents_6)
        self.notes_edit.setGeometry(QtCore.QRect(10, 10, 211, 191))
        self.notes_edit.setObjectName("notes_edit")
        self.dockWidget_6.setWidget(self.dockWidgetContents_6)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget_6)
        self.dockWidget_2 = QtWidgets.QDockWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dockWidget_2.sizePolicy().hasHeightForWidth())
        self.dockWidget_2.setSizePolicy(sizePolicy)
        self.dockWidget_2.setMinimumSize(QtCore.QSize(160, 93))
        self.dockWidget_2.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable|QtWidgets.QDockWidget.DockWidgetMovable)
        self.dockWidget_2.setObjectName("dockWidget_2")
        self.dockWidgetContents_2 = QtWidgets.QWidget()
        self.dockWidgetContents_2.setAutoFillBackground(True)
        self.dockWidgetContents_2.setObjectName("dockWidgetContents_2")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.dockWidgetContents_2)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.progressBar = QtWidgets.QProgressBar(self.dockWidgetContents_2)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout_4.addWidget(self.progressBar, 0, 0, 1, 1)
        self.progressBar_label = QtWidgets.QLabel(self.dockWidgetContents_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progressBar_label.sizePolicy().hasHeightForWidth())
        self.progressBar_label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setItalic(False)
        self.progressBar_label.setFont(font)
        self.progressBar_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.progressBar_label.setObjectName("progressBar_label")
        self.gridLayout_4.addWidget(self.progressBar_label, 1, 0, 2, 1)
        self.gridLayout_4.setRowStretch(0, 30)
        self.gridLayout_4.setRowStretch(1, 10)
        self.gridLayout_13.addLayout(self.gridLayout_4, 0, 0, 1, 1)
        self.dockWidget_2.setWidget(self.dockWidgetContents_2)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget_2)
        self.dockWidget_3 = QtWidgets.QDockWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dockWidget_3.sizePolicy().hasHeightForWidth())
        self.dockWidget_3.setSizePolicy(sizePolicy)
        self.dockWidget_3.setMinimumSize(QtCore.QSize(260, 160))
        self.dockWidget_3.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable|QtWidgets.QDockWidget.DockWidgetMovable)
        self.dockWidget_3.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        self.dockWidget_3.setObjectName("dockWidget_3")
        self.dockWidgetContents_3 = QtWidgets.QWidget()
        self.dockWidgetContents_3.setObjectName("dockWidgetContents_3")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.dockWidgetContents_3)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(10, 10, 208, 172))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.corrs_dropdown = QtWidgets.QComboBox(self.verticalLayoutWidget_2)
        self.corrs_dropdown.setObjectName("corrs_dropdown")
        self.corrs_dropdown.addItem("")
        self.corrs_dropdown.addItem("")
        self.corrs_dropdown.addItem("")
        self.corrs_dropdown.addItem("")
        self.corrs_dropdown.addItem("")
        self.verticalLayout_4.addWidget(self.corrs_dropdown)
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.calcCorrCurrentFile_button = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.calcCorrCurrentFile_button.setObjectName("calcCorrCurrentFile_button")
        self.gridLayout_7.addWidget(self.calcCorrCurrentFile_button, 4, 1, 1, 1)
        self.addToJoblist_button = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.addToJoblist_button.setObjectName("addToJoblist_button")
        self.gridLayout_7.addWidget(self.addToJoblist_button, 4, 0, 1, 1)
        self.precision_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.precision_label.setObjectName("precision_label")
        self.gridLayout_7.addWidget(self.precision_label, 0, 0, 1, 1)
        self.resolution_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.resolution_edit.setFont(font)
        self.resolution_edit.setObjectName("resolution_edit")
        self.gridLayout_7.addWidget(self.resolution_edit, 0, 1, 1, 1)
        self.chunkSize_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.chunkSize_label.setObjectName("chunkSize_label")
        self.gridLayout_7.addWidget(self.chunkSize_label, 1, 0, 1, 1)
        self.chunkSize_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.chunkSize_edit.setFont(font)
        self.chunkSize_edit.setObjectName("chunkSize_edit")
        self.gridLayout_7.addWidget(self.chunkSize_edit, 1, 1, 1, 1)
        self.algorithm_dropdown = QtWidgets.QComboBox(self.verticalLayoutWidget_2)
        self.algorithm_dropdown.setObjectName("algorithm_dropdown")
        self.algorithm_dropdown.addItem("")
        self.algorithm_dropdown.addItem("")
        self.algorithm_dropdown.addItem("")
        self.algorithm_dropdown.addItem("")
        self.gridLayout_7.addWidget(self.algorithm_dropdown, 2, 1, 1, 1)
        self.algorithm_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.algorithm_label.setObjectName("algorithm_label")
        self.gridLayout_7.addWidget(self.algorithm_label, 2, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label.setObjectName("label")
        self.gridLayout_7.addWidget(self.label, 3, 0, 1, 1)
        self.detector_dropdown = QtWidgets.QComboBox(self.verticalLayoutWidget_2)
        self.detector_dropdown.setObjectName("detector_dropdown")
        self.detector_dropdown.addItem("")
        self.detector_dropdown.addItem("")
        self.detector_dropdown.addItem("")
        self.detector_dropdown.addItem("")
        self.gridLayout_7.addWidget(self.detector_dropdown, 3, 1, 1, 1)
        self.gridLayout_7.setColumnStretch(0, 1)
        self.verticalLayout_4.addLayout(self.gridLayout_7)
        self.dockWidget_3.setWidget(self.dockWidgetContents_3)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dockWidget_3)
        self.dockWidget_4 = QtWidgets.QDockWidget(MainWindow)
        self.dockWidget_4.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dockWidget_4.sizePolicy().hasHeightForWidth())
        self.dockWidget_4.setSizePolicy(sizePolicy)
        self.dockWidget_4.setMinimumSize(QtCore.QSize(260, 475))
        self.dockWidget_4.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable|QtWidgets.QDockWidget.DockWidgetMovable)
        self.dockWidget_4.setObjectName("dockWidget_4")
        self.dockWidgetContents_4 = QtWidgets.QWidget()
        self.dockWidgetContents_4.setMinimumSize(QtCore.QSize(0, 450))
        self.dockWidgetContents_4.setObjectName("dockWidgetContents_4")
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(self.dockWidgetContents_4)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(10, 10, 231, 487))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.fitModel_dropdown = QtWidgets.QComboBox(self.verticalLayoutWidget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fitModel_dropdown.sizePolicy().hasHeightForWidth())
        self.fitModel_dropdown.setSizePolicy(sizePolicy)
        self.fitModel_dropdown.setMaximumSize(QtCore.QSize(16777215, 20))
        self.fitModel_dropdown.setObjectName("fitModel_dropdown")
        self.fitModel_dropdown.addItem("")
        self.fitModel_dropdown.addItem("")
        self.fitModel_dropdown.addItem("")
        self.fitModel_dropdown.addItem("")
        self.fitModel_dropdown.addItem("")
        self.verticalLayout_8.addWidget(self.fitModel_dropdown)
        self.verticalLayout_8.addSpacing(6)
        self.gridLayout_9 = QtWidgets.QGridLayout()
        self.gridLayout_9.setVerticalSpacing(6)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.fit2_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.fit2_edit.setFont(font)
        self.fit2_edit.setObjectName("fit2_edit")
        self.gridLayout_9.addWidget(self.fit2_edit, 2, 1, 1, 1)
        self.fit8_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.fit8_edit.setFont(font)
        self.fit8_edit.setObjectName("fit8_edit")
        self.gridLayout_9.addWidget(self.fit8_edit, 8, 1, 1, 1)
        self.fitstop_label = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.fitstop_label.setObjectName("fitstop_label")
        self.gridLayout_9.addWidget(self.fitstop_label, 13, 0, 1, 1)
        self.fitstart_label = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.fitstart_label.setObjectName("fitstart_label")
        self.gridLayout_9.addWidget(self.fitstart_label, 12, 0, 1, 1)
        self.fit5_checkBox = QtWidgets.QCheckBox(self.verticalLayoutWidget_4)
        self.fit5_checkBox.setObjectName("fit5_checkBox")
        self.gridLayout_9.addWidget(self.fit5_checkBox, 5, 0, 1, 1)
        self.fit0_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.fit0_edit.setFont(font)
        self.fit0_edit.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.fit0_edit.setObjectName("fit0_edit")
        self.gridLayout_9.addWidget(self.fit0_edit, 0, 1, 1, 1)
        self.fit7_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.fit7_edit.setFont(font)
        self.fit7_edit.setObjectName("fit7_edit")
        self.gridLayout_9.addWidget(self.fit7_edit, 7, 1, 1, 1)
        self.overwriteFit_button = QtWidgets.QPushButton(self.verticalLayoutWidget_4)
        self.overwriteFit_button.setIconSize(QtCore.QSize(0, 0))
        self.overwriteFit_button.setObjectName("overwriteFit_button")
        self.gridLayout_9.addWidget(self.overwriteFit_button, 14, 0, 1, 1)
        self.fit1_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.fit1_edit.setFont(font)
        self.fit1_edit.setObjectName("fit1_edit")
        self.gridLayout_9.addWidget(self.fit1_edit, 1, 1, 1, 1)
        self.fit7_checkBox = QtWidgets.QCheckBox(self.verticalLayoutWidget_4)
        self.fit7_checkBox.setObjectName("fit7_checkBox")
        self.gridLayout_9.addWidget(self.fit7_checkBox, 7, 0, 1, 1)
        self.fit5_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.fit5_edit.setFont(font)
        self.fit5_edit.setObjectName("fit5_edit")
        self.gridLayout_9.addWidget(self.fit5_edit, 5, 1, 1, 1)
        self.newFit_button = QtWidgets.QPushButton(self.verticalLayoutWidget_4)
        self.newFit_button.setObjectName("newFit_button")
        self.gridLayout_9.addWidget(self.newFit_button, 14, 1, 1, 1)
        self.fit6_checkBox = QtWidgets.QCheckBox(self.verticalLayoutWidget_4)
        self.fit6_checkBox.setObjectName("fit6_checkBox")
        self.gridLayout_9.addWidget(self.fit6_checkBox, 6, 0, 1, 1)
        self.fit8_checkBox = QtWidgets.QCheckBox(self.verticalLayoutWidget_4)
        self.fit8_checkBox.setObjectName("fit8_checkBox")
        self.gridLayout_9.addWidget(self.fit8_checkBox, 8, 0, 1, 1)
        self.fit4_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.fit4_edit.setFont(font)
        self.fit4_edit.setObjectName("fit4_edit")
        self.gridLayout_9.addWidget(self.fit4_edit, 4, 1, 1, 1)
        self.fit3_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.fit3_edit.setFont(font)
        self.fit3_edit.setObjectName("fit3_edit")
        self.gridLayout_9.addWidget(self.fit3_edit, 3, 1, 1, 1)
        self.fit4_checkBox = QtWidgets.QCheckBox(self.verticalLayoutWidget_4)
        self.fit4_checkBox.setObjectName("fit4_checkBox")
        self.gridLayout_9.addWidget(self.fit4_checkBox, 4, 0, 1, 1)
        self.fit6_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.fit6_edit.setFont(font)
        self.fit6_edit.setObjectName("fit6_edit")
        self.gridLayout_9.addWidget(self.fit6_edit, 6, 1, 1, 1)
        self.fit9_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.fit9_edit.setFont(font)
        self.fit9_edit.setObjectName("fit9_edit")
        self.gridLayout_9.addWidget(self.fit9_edit, 9, 1, 1, 1)
        self.fit9_checkBox = QtWidgets.QCheckBox(self.verticalLayoutWidget_4)
        self.fit9_checkBox.setObjectName("fit9_checkBox")
        self.gridLayout_9.addWidget(self.fit9_checkBox, 9, 0, 1, 1)
        self.fit10_checkBox = QtWidgets.QCheckBox(self.verticalLayoutWidget_4)
        self.fit10_checkBox.setObjectName("fit10_checkBox")
        self.gridLayout_9.addWidget(self.fit10_checkBox, 10, 0, 1, 1)
        self.fit10_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.fit10_edit.setFont(font)
        self.fit10_edit.setObjectName("fit10_edit")
        self.gridLayout_9.addWidget(self.fit10_edit, 10, 1, 1, 1)
        self.fit1_checkBox = QtWidgets.QCheckBox(self.verticalLayoutWidget_4)
        self.fit1_checkBox.setObjectName("fit1_checkBox")
        self.gridLayout_9.addWidget(self.fit1_checkBox, 1, 0, 1, 1)
        self.fitstart_spinBox = QtWidgets.QSpinBox(self.verticalLayoutWidget_4)
        self.fitstart_spinBox.setMinimum(1)
        self.fitstart_spinBox.setMaximum(999)
        self.fitstart_spinBox.setObjectName("fitstart_spinBox")
        self.gridLayout_9.addWidget(self.fitstart_spinBox, 12, 1, 1, 1)
        self.fitstop_spinBox = QtWidgets.QSpinBox(self.verticalLayoutWidget_4)
        self.fitstop_spinBox.setMinimum(1)
        self.fitstop_spinBox.setMaximum(10000)
        self.fitstop_spinBox.setProperty("value", 100)
        self.fitstop_spinBox.setObjectName("fitstop_spinBox")
        self.gridLayout_9.addWidget(self.fitstop_spinBox, 13, 1, 1, 1)
        self.fit0_checkBox = QtWidgets.QCheckBox(self.verticalLayoutWidget_4)
        self.fit0_checkBox.setObjectName("fit0_checkBox")
        self.gridLayout_9.addWidget(self.fit0_checkBox, 0, 0, 1, 1)
        self.fit3_checkBox = QtWidgets.QCheckBox(self.verticalLayoutWidget_4)
        self.fit3_checkBox.setObjectName("fit3_checkBox")
        self.gridLayout_9.addWidget(self.fit3_checkBox, 3, 0, 1, 1)
        self.fit2_checkBox = QtWidgets.QCheckBox(self.verticalLayoutWidget_4)
        self.fit2_checkBox.setObjectName("fit2_checkBox")
        self.gridLayout_9.addWidget(self.fit2_checkBox, 2, 0, 1, 1)
        self.fitweighted_checkBox = QtWidgets.QCheckBox(self.verticalLayoutWidget_4)
        self.fitweighted_checkBox.setObjectName("fitweighted_checkBox")
        self.gridLayout_9.addWidget(self.fitweighted_checkBox, 11, 0, 1, 1)
        self.verticalLayout_8.addLayout(self.gridLayout_9)
        self.dockWidget_4.setWidget(self.dockWidgetContents_4)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dockWidget_4)
        self.dockWidget_5 = QtWidgets.QDockWidget(MainWindow)
        self.dockWidget_5.setMinimumSize(QtCore.QSize(270, 230))
        self.dockWidget_5.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable|QtWidgets.QDockWidget.DockWidgetMovable)
        self.dockWidget_5.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        self.dockWidget_5.setObjectName("dockWidget_5")
        self.dockWidgetContents_5 = QtWidgets.QWidget()
        self.dockWidgetContents_5.setObjectName("dockWidgetContents_5")
        self.gridLayoutWidget = QtWidgets.QWidget(self.dockWidgetContents_5)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 0, 251, 210))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout_10.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.visc_edit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.visc_edit.setFont(font)
        self.visc_edit.setObjectName("visc_edit")
        self.gridLayout_10.addWidget(self.visc_edit, 5, 1, 1, 1)
        self.viscosity_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.viscosity_label.setObjectName("viscosity_label")
        self.gridLayout_10.addWidget(self.viscosity_label, 5, 0, 1, 1)
        self.w0_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.w0_label.setObjectName("w0_label")
        self.gridLayout_10.addWidget(self.w0_label, 0, 0, 1, 1)
        self.D_edit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.D_edit.setFont(font)
        self.D_edit.setObjectName("D_edit")
        self.gridLayout_10.addWidget(self.D_edit, 2, 1, 1, 1)
        self.D_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.D_label.setObjectName("D_label")
        self.gridLayout_10.addWidget(self.D_label, 2, 0, 1, 1)
        self.w0_edit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.w0_edit.setFont(font)
        self.w0_edit.setObjectName("w0_edit")
        self.gridLayout_10.addWidget(self.w0_edit, 0, 1, 1, 1)
        self.T_edit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.T_edit.setFont(font)
        self.T_edit.setObjectName("T_edit")
        self.gridLayout_10.addWidget(self.T_edit, 4, 1, 1, 1)
        self.temperature_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.temperature_label.setObjectName("temperature_label")
        self.gridLayout_10.addWidget(self.temperature_label, 4, 0, 1, 1)
        self.keepFixed_dropdown = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.keepFixed_dropdown.setObjectName("keepFixed_dropdown")
        self.keepFixed_dropdown.addItem("")
        self.keepFixed_dropdown.addItem("")
        self.gridLayout_10.addWidget(self.keepFixed_dropdown, 3, 0, 1, 1)
        self.updateDiffLaw_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.updateDiffLaw_button.setObjectName("updateDiffLaw_button")
        self.gridLayout_10.addWidget(self.updateDiffLaw_button, 3, 1, 1, 1)
        self.calcDiameter_dropdown = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.calcDiameter_dropdown.setObjectName("calcDiameter_dropdown")
        self.calcDiameter_dropdown.addItem("")
        self.calcDiameter_dropdown.addItem("")
        self.calcDiameter_dropdown.addItem("")
        self.gridLayout_10.addWidget(self.calcDiameter_dropdown, 7, 0, 1, 1)
        self.diameter_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.diameter_label.setObjectName("diameter_label")
        self.gridLayout_10.addWidget(self.diameter_label, 6, 0, 1, 1)
        self.diameter_edit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.diameter_edit.setFont(font)
        self.diameter_edit.setObjectName("diameter_edit")
        self.gridLayout_10.addWidget(self.diameter_edit, 6, 1, 1, 1)
        self.updateDiffLawDiameter_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.updateDiffLawDiameter_button.setObjectName("updateDiffLawDiameter_button")
        self.gridLayout_10.addWidget(self.updateDiffLawDiameter_button, 7, 1, 1, 1)
        self.dockWidget_5.setWidget(self.dockWidgetContents_5)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dockWidget_5)
        self.actionOpen_image = QtWidgets.QAction(MainWindow)
        self.actionOpen_image.setObjectName("actionOpen_image")
        self.actionChange_image = QtWidgets.QAction(MainWindow)
        self.actionChange_image.setObjectName("actionChange_image")
        self.actionOpen_session = QtWidgets.QAction(MainWindow)
        self.actionOpen_session.setObjectName("actionOpen_session")
        self.actionSave_session = QtWidgets.QAction(MainWindow)
        self.actionSave_session.setObjectName("actionSave_session")
        self.actionNew_session = QtWidgets.QAction(MainWindow)
        self.actionNew_session.setObjectName("actionNew_session")
        self.actionCorrelation = QtWidgets.QAction(MainWindow)
        self.actionCorrelation.setObjectName("actionCorrelation")
        self.actionAllCorrelationsCurrentFile = QtWidgets.QAction(MainWindow)
        self.actionAllCorrelationsCurrentFile.setObjectName("actionAllCorrelationsCurrentFile")
        self.actionCurrent_fit = QtWidgets.QAction(MainWindow)
        self.actionCurrent_fit.setObjectName("actionCurrent_fit")
        self.actionCurrent_correlation = QtWidgets.QAction(MainWindow)
        self.actionCurrent_correlation.setObjectName("actionCurrent_correlation")
        self.det_Genoa_Instrument_5x5 = QtWidgets.QAction(MainWindow)
        self.det_Genoa_Instrument_5x5.setCheckable(True)
        self.det_Genoa_Instrument_5x5.setChecked(True)
        self.det_Genoa_Instrument_5x5.setObjectName("det_Genoa_Instrument_5x5")
        self.det_Genoa_Instruments_7x7 = QtWidgets.QAction(MainWindow)
        self.det_Genoa_Instruments_7x7.setCheckable(True)
        self.det_Genoa_Instruments_7x7.setObjectName("det_Genoa_Instruments_7x7")
        self.det_Nikon_NSPARC = QtWidgets.QAction(MainWindow)
        self.det_Nikon_NSPARC.setCheckable(True)
        self.det_Nikon_NSPARC.setObjectName("det_Nikon_NSPARC")
        self.det_PI_Imaging_23 = QtWidgets.QAction(MainWindow)
        self.det_PI_Imaging_23.setCheckable(True)
        self.det_PI_Imaging_23.setObjectName("det_PI_Imaging_23")
        self.actionSave_session_as = QtWidgets.QAction(MainWindow)
        self.actionSave_session_as.setObjectName("actionSave_session_as")
        self.actionCurrent_FFS_file = QtWidgets.QAction(MainWindow)
        self.actionCurrent_FFS_file.setObjectName("actionCurrent_FFS_file")
        self.actionCopy_current_fit_as_data = QtWidgets.QAction(MainWindow)
        self.actionCopy_current_fit_as_data.setObjectName("actionCopy_current_fit_as_data")
        self.actionCreate_Jupyter_Notebook = QtWidgets.QAction(MainWindow)
        self.actionCreate_Jupyter_Notebook.setObjectName("actionCreate_Jupyter_Notebook")
        self.actionCopy_current_correlation = QtWidgets.QAction(MainWindow)
        self.actionCopy_current_correlation.setObjectName("actionCopy_current_correlation")
        self.actionFilter_out_bad_chunks = QtWidgets.QAction(MainWindow)
        self.actionFilter_out_bad_chunks.setObjectName("actionFilter_out_bad_chunks")
        self.actionCurrent_Image = QtWidgets.QAction(MainWindow)
        self.actionCurrent_Image.setObjectName("actionCurrent_Image")
        self.actionPlot_in_Jupyter_Notebook = QtWidgets.QAction(MainWindow)
        self.actionPlot_in_Jupyter_Notebook.setObjectName("actionPlot_in_Jupyter_Notebook")
        self.actionExport_results_to_Excel = QtWidgets.QAction(MainWindow)
        self.actionExport_results_to_Excel.setObjectName("actionExport_results_to_Excel")
        self.actionExport_current_correlation_to_Excel = QtWidgets.QAction(MainWindow)
        self.actionExport_current_correlation_to_Excel.setObjectName("actionExport_current_correlation_to_Excel")
        self.menuFile.addAction(self.actionOpen_image)
        self.menuFile.addAction(self.actionChange_image)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionOpen_session)
        self.menuFile.addAction(self.actionSave_session)
        self.menuFile.addAction(self.actionSave_session_as)
        self.menuFile.addAction(self.actionNew_session)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExport_results_to_Excel)
        self.menuFile.addAction(self.actionExport_current_correlation_to_Excel)
        self.menuCalculate.addAction(self.actionCorrelation)
        self.menuCalculate.addAction(self.actionAllCorrelationsCurrentFile)
        self.menuRemove.addAction(self.actionCurrent_fit)
        self.menuRemove.addAction(self.actionCurrent_correlation)
        self.menuRemove.addAction(self.actionCurrent_FFS_file)
        self.menuRemove.addAction(self.actionCurrent_Image)
        self.menuTools.addAction(self.actionCopy_current_correlation)
        self.menuTools.addAction(self.actionCopy_current_fit_as_data)
        self.menuTools.addSeparator()
        self.menuTools.addAction(self.actionPlot_in_Jupyter_Notebook)
        self.menuTools.addAction(self.actionCreate_Jupyter_Notebook)
        self.menuTools.addSeparator()
        self.menuTools.addAction(self.actionFilter_out_bad_chunks)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuCalculate.menuAction())
        self.menubar.addAction(self.menuRemove.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())

        self.retranslateUi(MainWindow)
        self.showElements_widget.setCurrentRow(2)
        self.apply_professional_style(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def apply_professional_style(self, MainWindow):
        """Apply a modern visual theme without changing widget behavior."""
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setStyle("Fusion")
            self._menu_polisher = _MenuPolisher(MainWindow)
            app.installEventFilter(self._menu_polisher)

        self._ui_scale = _screen_ui_scale(MainWindow)
        # General controls scale only slightly with DPI. Fit controls are
        # clamped separately below so they do not become too tall on laptops.
        self._button_height = _scaled(23, self._ui_scale)
        self._field_height = _scaled(23, self._ui_scale)
        # Closed combo boxes are kept more compact than normal line edits.
        # On some high-DPI laptops Qt makes combo boxes noticeably taller
        # than line edits unless their maximum height is clamped separately.
        # Dense right-side controls should not follow the laptop DPI scale.
        # With a laptop attached to an external monitor, Qt/Windows can report
        # a larger logical DPI, which made combo boxes too tall. Use compact
        # fixed logical-pixel heights for form fields and dropdowns.
        self._combo_height = 19
        self._fit_field_height = 21
        self._fit_combo_height = 19
        self._top_button_height = _scaled(24, min(self._ui_scale, 1.05))
        self._nav_icon_size = _scaled(16, min(self._ui_scale, 1.05))

        MainWindow.setMinimumSize(QtCore.QSize(1050, 720))
        MainWindow.setDockOptions(
            QtWidgets.QMainWindow.AnimatedDocks
            | QtWidgets.QMainWindow.AllowNestedDocks
            | QtWidgets.QMainWindow.AllowTabbedDocks
        )

        base_font = QtGui.QFont("Segoe UI")
        base_font.setPointSizeF(9.0 * self._ui_scale)
        base_font.setStyleHint(QtGui.QFont.SansSerif)
        MainWindow.setFont(base_font)

        # Improve spacing in the main workspace.
        self.gridLayout_8.setContentsMargins(9, 9, 9, 9)
        self.gridLayout_8.setHorizontalSpacing(6)
        self.gridLayout_8.setVerticalSpacing(6)
        self.central_window.setSpacing(6)
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_4.setSpacing(6)
        self.horizontalLayout_5.setSpacing(6)
        self.gridLayout_6.setHorizontalSpacing(6)
        self.horizontalLayout_7.setSpacing(6)

        # Make dock contents responsive instead of relying on fixed geometry.
        responsive_docks = (
            (self.dockWidgetContents_6, self.notes_edit),
            (self.dockWidgetContents_3, self.verticalLayoutWidget_2),
            (self.dockWidgetContents_4, self.verticalLayoutWidget_4),
            (self.dockWidgetContents_5, self.gridLayoutWidget),
        )
        for container, child in responsive_docks:
            if container.layout() is None:
                layout = QtWidgets.QVBoxLayout(container)
                layout.setContentsMargins(6, 6, 6, 6)
                layout.setSpacing(4)
                layout.addWidget(child)

        # Keep the correlation-calculation controls grouped at the top when
        # the dock is stretched vertically. The stretch absorbs extra space
        # below the form instead of between the selector and the fields.
        self.verticalLayout_4.setSpacing(4)
        self.verticalLayout_4.setAlignment(QtCore.Qt.AlignTop)
        self.gridLayout_7.setVerticalSpacing(4)
        self.gridLayout_7.setAlignment(QtCore.Qt.AlignTop)
        self.verticalLayout_4.addStretch(1)
        self.verticalLayoutWidget_2.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )

        # Keep the Fit analysis controls grouped at the top when its dock is
        # stretched. Extra height is absorbed below the complete form.
        self.verticalLayout_8.setSpacing(_scaled(4, self._ui_scale))
        self.verticalLayout_8.setAlignment(QtCore.Qt.AlignTop)
        self.gridLayout_9.setVerticalSpacing(_scaled(6, self._ui_scale))
        self.gridLayout_9.setHorizontalSpacing(_scaled(6, self._ui_scale))
        self.gridLayout_9.setAlignment(QtCore.Qt.AlignTop)
        self.verticalLayout_8.addStretch(1)
        self.verticalLayoutWidget_4.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )

        # Do the same for Diffusion analysis. This panel uses a grid directly,
        # so an expanding spacer row keeps all controls together at the top.
        self.gridLayout_10.setVerticalSpacing(4)
        self.gridLayout_10.setAlignment(QtCore.Qt.AlignTop)
        self._diffusion_bottom_spacer = QtWidgets.QSpacerItem(
            0, 0,
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.gridLayout_10.addItem(self._diffusion_bottom_spacer, 8, 0, 1, 2)
        self.gridLayout_10.setRowStretch(8, 1)
        self.gridLayoutWidget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )

        # Visual roles used by the stylesheet.
        primary_buttons = (
            self.saveLabel_button,
            self.calcCorrCurrentFile_button,
            self.newFit_button,
            self.updateDiffLaw_button,
            self.updateDiffLawDiameter_button,
        )
        for button in primary_buttons:
            button.setProperty("role", "primary")

        secondary_buttons = (
            self.addToJoblist_button,
            self.overwriteFit_button,
            self.imageName_button,
        )
        for button in secondary_buttons:
            button.setProperty("role", "secondary")

        navigation_buttons = (
            self.prevFCSfile_button,
            self.nextFCSfile_button,
            self.prevImage_button,
            self.nextImage_button,
        )
        for button in navigation_buttons:
            button.setProperty("role", "navigation")
            button.setMinimumWidth(_scaled(34, self._ui_scale))
            button.setMaximumWidth(_scaled(42, self._ui_scale))
            button.setMinimumHeight(self._top_button_height)
            button.setIconSize(QtCore.QSize(self._nav_icon_size, self._nav_icon_size))

        file_buttons = (
            self.FCSfile0_button,
            self.FCSfile1_button,
            self.FCSfile2_button,
            self.FCSfile3_button,
            self.FCSfile4_button,
        )
        for button in file_buttons:
            button.setProperty("role", "fileTab")
            button.setMinimumHeight(self._top_button_height)
            # Keep one file tab visibly active without changing its click signal.
            button.setCheckable(True)
            button.setAutoExclusive(True)

        # File-tab colors are painted by _FileTabButton, which avoids native
        # platform styles replacing the backgrounds with grey.

        # The first slot is the active file when the interface is created.
        self.FCSfile0_button.setChecked(True)

        plot_widgets = (
            self.image_widget,
            self.fingerprint_widget,
            self.timetrace_widget,
            self.correlations_widget,
            self.difflaw_widget,
        )
        for widget in plot_widgets:
            widget.setProperty("panel", "plot")

        self.FCSFileName_label.setProperty("role", "title")
        self.FCSFolderName_label.setProperty("role", "subtitle")
        self.imageInfo_label.setWordWrap(True)
        self.progressBar.setTextVisible(True)
        self.notes_edit.setPlaceholderText("Add notes about the current measurement...")

        # Small usability improvements that do not alter signals or data flow.
        for button in MainWindow.findChildren(QtWidgets.QPushButton):
            button.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            button.setMinimumHeight(self._button_height)
            button_font = button.font()
            button_font.setBold(False)
            button_font.setWeight(QtGui.QFont.Normal)
            button.setFont(button_font)

        for combo in MainWindow.findChildren(QtWidgets.QComboBox):
            combo.setMinimumHeight(self._combo_height)
            combo.setMaximumHeight(self._combo_height + 1)
            combo.setMaxVisibleItems(max(8, min(25, combo.count())))
            combo.view().setTextElideMode(QtCore.Qt.ElideRight)
            combo.view().setIconSize(QtCore.QSize(0, 0))
            combo.view().setItemDelegate(_CleanComboItemDelegate(combo.view()))
            if hasattr(combo.view(), "setIndentation"):
                combo.view().setIndentation(0)
            combo.view().setStyleSheet(
                "QAbstractItemView {"
                "  background: #ffffff;"
                "  border: 1px solid #cfd8e6;"
                "  border-radius: 3px;"
                "  padding: 2px;"
                "  outline: none;"
                "}"
                "QAbstractItemView::item {"
                "  min-height: 20px;"
                "  padding: 1px 7px;"
                "  border: none;"
                "  border-radius: 3px;"
                "}"
                "QAbstractItemView::item:hover {"
                "  background: #f1f5f9;"
                "  color: #172033;"
                "  border: none;"
                "}"
                "QAbstractItemView::item:selected {"
                "  background: #e8f5ee;"
                "  color: #166534;"
                "  border: none;"
                "}"
                "QAbstractItemView::indicator {"
                "  width: 0px;"
                "  height: 0px;"
                "  image: none;"
                "}"
            )

        # The two central analysis selectors contain longer descriptions.
        # Set the popup row geometry explicitly; stylesheet min-height alone
        # only enlarges the painted selection rectangle on some Qt styles.
        self._central_combo_delegates = []
        for combo in (self.showchunkscorr_dropdown, self.difflaw_dropdown):
            view = combo.view()
            delegate = _CleanComboItemDelegate(view, minimum_height=24)
            view.setItemDelegate(delegate)
            self._central_combo_delegates.append(delegate)

            view.setUniformItemSizes(False)
            if hasattr(view, "setSpacing"):
                view.setSpacing(1)

            # SizeHintRole forces the actual popup rows to become taller.
            # It does not change the height of the closed combo box itself.
            font_metrics = combo.fontMetrics()
            for row in range(combo.count()):
                text_width = font_metrics.horizontalAdvance(combo.itemText(row)) + 30
                row_width = max(combo.sizeHint().width(), text_width)
                combo.setItemData(
                    row,
                    QtCore.QSize(row_width, 24),
                    QtCore.Qt.SizeHintRole,
                )

            view.setStyleSheet(
                "QAbstractItemView {"
                "  background: #ffffff;"
                "  border: 1px solid #cfd8e6;"
                "  border-radius: 3px;"
                "  padding: 2px;"
                "  outline: none;"
                "}"
                "QAbstractItemView::item {"
                "  padding: 1px 7px;"
                "  border: none;"
                "  border-radius: 3px;"
                "}"
                "QAbstractItemView::item:hover {"
                "  background: #f1f5f9;"
                "  color: #172033;"
                "}"
                "QAbstractItemView::item:selected {"
                "  background: #e8f5ee;"
                "  color: #166534;"
                "}"
                "QAbstractItemView::indicator {"
                "  width: 0px;"
                "  height: 0px;"
                "  image: none;"
                "}"
            )

        for edit in MainWindow.findChildren(QtWidgets.QLineEdit):
            edit.setMinimumHeight(23)

        # Allow the image placeholder text to fit comfortably on two lines.
        self.imageName_button.setMinimumHeight(42)

        # Align the fit-range labels with the neighboring spin boxes.
        for label in (self.fitstart_label, self.fitstop_label):
            label.setMinimumHeight(23)
            label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            label.setTextFormat(QtCore.Qt.PlainText)

        for spinbox in (self.fitstart_spinBox, self.fitstop_spinBox, self.chunk_spinBox):
            spinbox.setMinimumHeight(23)
            spinbox.lineEdit().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # Compact the Fit analysis controls explicitly. The general DPI-aware
        # stylesheet keeps the GUI readable, but these controls should stay
        # compact because the dock contains many rows.
        fit_edits = tuple(
            getattr(self, f"fit{index}_edit")
            for index in range(11)
        )
        for edit in fit_edits:
            edit.setProperty("fitValue", True)
            edit.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Fixed,
            )
            edit.setFixedHeight(self._fit_field_height)

        fit_checks = tuple(
            getattr(self, f"fit{index}_checkBox")
            for index in range(11)
        ) + (self.fitweighted_checkBox,)
        for checkbox in fit_checks:
            checkbox.setMinimumHeight(self._fit_field_height)
            checkbox.setMaximumHeight(self._fit_field_height + 2)

        for spinbox in (self.fitstart_spinBox, self.fitstop_spinBox):
            spinbox.setMinimumHeight(self._fit_field_height)
            spinbox.setMaximumHeight(self._fit_field_height + 1)

        self.fitModel_dropdown.setMinimumHeight(self._fit_combo_height)
        self.fitModel_dropdown.setMaximumHeight(self._fit_combo_height + 1)
        self.gridLayout_9.setVerticalSpacing(3)

        # Apply the same compact treatment to the Calculate correlations
        # and Diffusion analysis panels. These right-side docks contain
        # dense forms, so they should use the compact field height instead
        # of the general screen-scaled field height. This prevents tall
        # dropdowns/text boxes on high-DPI laptop screens while preserving
        # readability on the desktop screen.
        self.gridLayout_7.setVerticalSpacing(3)
        self.gridLayout_7.setHorizontalSpacing(_scaled(6, min(self._ui_scale, 1.05)))
        self.verticalLayout_4.setSpacing(_scaled(3, min(self._ui_scale, 1.05)))

        compact_corr_fields = (
            self.resolution_edit,
            self.chunkSize_edit,
            self.algorithm_dropdown,
            self.detector_dropdown,
            self.corrs_dropdown,
        )
        for widget in compact_corr_fields:
            widget.setMinimumHeight(self._fit_field_height)
            widget.setMaximumHeight(self._fit_field_height + 1)
        for combo in (self.corrs_dropdown, self.algorithm_dropdown, self.detector_dropdown):
            combo.setMinimumHeight(self._combo_height)
            combo.setMaximumHeight(self._combo_height + 1)

        for button in (self.addToJoblist_button, self.calcCorrCurrentFile_button):
            button.setMinimumHeight(self._fit_field_height + 1)
            button.setMaximumHeight(self._fit_field_height + 3)

        self.gridLayout_10.setVerticalSpacing(3)
        self.gridLayout_10.setHorizontalSpacing(_scaled(6, min(self._ui_scale, 1.05)))

        compact_diff_fields = (
            self.w0_edit,
            self.D_edit,
            self.T_edit,
            self.visc_edit,
            self.diameter_edit,
            self.keepFixed_dropdown,
            self.calcDiameter_dropdown,
        )
        for widget in compact_diff_fields:
            widget.setMinimumHeight(self._fit_field_height)
            widget.setMaximumHeight(self._fit_field_height + 1)
        for combo in (self.keepFixed_dropdown, self.calcDiameter_dropdown):
            combo.setMinimumHeight(self._combo_height)
            combo.setMaximumHeight(self._combo_height + 1)

        for button in (self.updateDiffLaw_button, self.updateDiffLawDiameter_button):
            button.setMinimumHeight(self._fit_field_height + 1)
            button.setMaximumHeight(self._fit_field_height + 3)

        self.xcoord_edit.setMaximumWidth(80)
        self.ycoord_edit.setMaximumWidth(_scaled(80, self._ui_scale))
        self.saveLabel_button.setMaximumWidth(_scaled(70, self._ui_scale))
        self.showElements_widget.setMinimumSize(QtCore.QSize(120, 58))
        self.showElements_widget.setMaximumSize(QtCore.QSize(120, 58))

        self.correlations_treeWidget.setAlternatingRowColors(True)
        self.correlations_treeWidget.setAnimated(False)
        self.correlations_treeWidget.setUniformRowHeights(True)
        self.correlations_treeWidget.header().setStretchLastSection(True)
        self.correlations_treeWidget.header().setStyleSheet(
            "QHeaderView::section { font-weight: 400; }"
        )

        # Custom-drawn arrows avoid platform icon padding differences.
        # This keeps the left and right arrows the same visual size on all screens.
        style = MainWindow.style()
        self.prevFCSfile_button.setIcon(_make_arrow_icon("left"))
        self.nextFCSfile_button.setIcon(_make_arrow_icon("right"))
        self.prevImage_button.setIcon(_make_arrow_icon("left"))
        self.nextImage_button.setIcon(_make_arrow_icon("right"))
        for button in (
            self.prevFCSfile_button,
            self.nextFCSfile_button,
            self.prevImage_button,
            self.nextImage_button,
        ):
            button.setIconSize(QtCore.QSize(self._nav_icon_size, self._nav_icon_size))
        # Keep the compact OK button text-only; no redundant check icon.
        self.saveLabel_button.setIcon(QtGui.QIcon())
        self.actionOpen_image.setIcon(style.standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        self.actionOpen_session.setIcon(style.standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        self.actionSave_session.setIcon(style.standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        self.actionSave_session_as.setIcon(style.standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        self.actionNew_session.setIcon(style.standardIcon(QtWidgets.QStyle.SP_FileIcon))
        self.actionCurrent_fit.setIcon(style.standardIcon(QtWidgets.QStyle.SP_TrashIcon))
        self.actionCurrent_correlation.setIcon(style.standardIcon(QtWidgets.QStyle.SP_TrashIcon))
        self.actionCurrent_FFS_file.setIcon(style.standardIcon(QtWidgets.QStyle.SP_TrashIcon))
        self.actionCurrent_Image.setIcon(style.standardIcon(QtWidgets.QStyle.SP_TrashIcon))

        MainWindow.setStyleSheet(
            """
            QMainWindow {
                background: #f0f0f0;
                color: #172033;
            }

            QWidget#centralwidget {
                background: #fafafa;
            }

            QWidget {
                color: #172033;
                font-family: "Segoe UI", "Inter", sans-serif;
                font-size: 9pt;
            }

            QLabel[role="title"] {
                color: #111827;
                font-size: 11pt;
                font-weight: 400;
            }

            QLabel[role="subtitle"] {
                color: #6b7280;
                font-size: 8.5pt;
            }

            QMenuBar {
                background: #fafafa;
                color: #263244;
                border: none;
                border-bottom: 1px solid #e2e7ef;
                padding: 2px 5px;
                spacing: 1px;
            }

            QMenuBar::item {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 3px 8px;
                margin: 0px;
            }

            QMenuBar::item:selected {
                background: #f0f3f9;
                border-color: #dce3ee;
                color: #25324a;
            }

            QMenuBar::item:pressed {
                background: #e8edf6;
                border-color: #cfd8e6;
                color: #1f2a44;
            }

            QMenu {
                background: #ffffff;
                color: #263244;
                border: 1px solid #d7deea;
                border-radius: 6px;
                padding: 3px;
            }

            QMenu::item {
                background: transparent;
                border: none;
                border-radius: 4px;
                padding: 4px 20px 4px 9px;
                margin: 0px;
            }

            QMenu::item:selected {
                background: #eef2ff;
                color: #3730a3;
            }

            QMenu::item:disabled {
                color: #a0a8b5;
                background: transparent;
            }

            QMenu::separator {
                height: 1px;
                background: #e7ebf1;
                margin: 3px 6px;
            }

            QMenu::icon {
                width: 0px;
                height: 0px;
            }

            QMenu::indicator {
                width: 0px;
                height: 0px;
            }

            QPushButton {
                background: #ffffff;
                border: 1px solid #cfd8e6;
                border-radius: 5px;
                padding: 3px 8px;
                font-weight: 400;
            }

            QPushButton:hover {
                background: #f0f3ff;
                border-color: #818cf8;
                color: #3730a3;
            }

            QPushButton:pressed {
                background: #e0e7ff;
                border-color: #6366f1;
            }

            QPushButton:disabled {
                background: #eef1f5;
                color: #9aa4b2;
                border-color: #dce2ea;
            }

            QPushButton[role="primary"] {
                background: #4f46e5;
                color: #ffffff;
                border: 1px solid #4f46e5;
                font-weight: 400;
            }

            QPushButton[role="primary"]:hover {
                background: #4338ca;
                border-color: #4338ca;
                color: #ffffff;
            }

            QPushButton[role="primary"]:pressed {
                background: #3730a3;
                border-color: #3730a3;
            }

            QPushButton[role="secondary"] {
                background: #eef2ff;
                color: #3730a3;
                border-color: #c7d2fe;
                font-weight: 400;
            }

            QPushButton[role="secondary"]:hover {
                background: #dfe5ff;
                color: #312e81;
                border-color: #6366f1;
            }

            QPushButton[role="secondary"]:pressed {
                background: #cfd7ff;
                color: #312e81;
                border-color: #4f46e5;
            }

            QPushButton[role="navigation"] {
                background: #e9edf5;
                border-color: #d5dce8;
                border-radius: 6px;
                padding: 2px;
                font-weight: 400;
            }

            QPushButton[role="navigation"]:hover {
                background: #dfe5ff;
                border-color: #6366f1;
            }

            QPushButton[role="navigation"]:pressed {
                background: #cfd7ff;
                border-color: #4f46e5;
            }

            QPushButton[role="fileTab"] {
                background: #eef2ff;
                color: #3730a3;
                border: 1px solid #c7d2fe;
                padding-left: 9px;
                padding-right: 9px;
                font-weight: 400;
            }

            QPushButton[role="fileTab"]:hover {
                background: #dfe5ff;
                color: #312e81;
                border-color: #6366f1;
            }

            QPushButton[role="fileTab"]:pressed {
                background: #cfd7ff;
                color: #312e81;
                border-color: #4f46e5;
            }

            QPushButton[role="fileTab"]:checked {
                background: #4f46e5;
                color: #ffffff;
                border: 1px solid #4f46e5;
            }

            QPushButton[role="fileTab"]:checked:hover {
                background: #4338ca;
                color: #ffffff;
                border-color: #4338ca;
            }

            QPushButton[role="fileTab"]:checked:pressed {
                background: #3730a3;
                color: #ffffff;
                border-color: #3730a3;
            }

            QPushButton[role="fileTab"]:focus {
                outline: none;
            }

            QLineEdit, QComboBox, QSpinBox, QPlainTextEdit {
                background: #ffffff;
                border: 1px solid #cfd8e6;
                border-radius: 5px;
                padding: 2px 6px;
                selection-background-color: #6366f1;
                selection-color: #ffffff;
            }

            QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QPlainTextEdit:hover {
                border-color: #a5b4fc;
            }

            /* Fit value fields keep one exact outer height on every style. */
            QLineEdit[fitValue="true"] {
                min-height: 21px;
                max-height: 21px;
                padding: 0px 6px;
            }

            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus {
                border: 1px solid #6366f1;
                padding: 2px 6px;
            }

            /* Match unavailable fit values with their greyed-out "None"
               labels without disabling the line edits or changing behavior. */
            QLineEdit[fitUnavailable="true"] {
                color: #9ca3af;
                background: #f8fafc;
                border-color: #d9dee7;
            }

            QLineEdit[fitUnavailable="true"]:hover,
            QLineEdit[fitUnavailable="true"]:focus {
                color: #9ca3af;
                background: #f8fafc;
                border-color: #cbd5e1;
            }

            QComboBox {
                padding: 1px 6px;
            }

            QComboBox::drop-down {
                border: none;
                width: 18px;
            }

            QComboBox QAbstractItemView {
                background: #ffffff;
                border: 1px solid #cfd8e6;
                border-radius: 5px;
                padding: 2px;
                selection-background-color: #e8f5ee;
                selection-color: #166534;
                outline: none;
            }

            QComboBox QAbstractItemView::item {
                border: none;
                border-radius: 3px;
                padding-left: 7px;
            }

            QComboBox QAbstractItemView::indicator {
                width: 0px;
                height: 0px;
                image: none;
                border: none;
            }

            QSpinBox#fitstart_spinBox,
            QSpinBox#fitstop_spinBox,
            QSpinBox#chunk_spinBox {
                padding: 0px 6px 0px 6px;
            }

            QSpinBox#fitstart_spinBox:focus,
            QSpinBox#fitstop_spinBox:focus,
            QSpinBox#chunk_spinBox:focus {
                padding: 0px 6px 0px 6px;
            }

            QCheckBox {
                spacing: 4px;
                padding: 0px;
            }

            /* Fit parameters that are unavailable are disabled by the
               application. Restore a clear muted appearance for labels such
               as "None", which the global QWidget color would otherwise
               override. */
            QCheckBox:disabled {
                color: #9ca3af;
            }

            /* Disabled controls for real fit parameters keep a readable
               parameter name. Only their disabled indicator is muted. */
            QCheckBox[fitParameter="true"]:disabled {
                color: #172033;
            }

            QCheckBox::indicator {
                width: 13px;
                height: 13px;
            }

            QCheckBox::indicator:disabled {
                border-color: #cbd5e1;
                background: #f1f5f9;
            }

            QTreeWidget, QListWidget {
                background: #ffffff;
                alternate-background-color: #f7f9fc;
                border: 1px solid #d7deea;
                border-radius: 6px;
                outline: none;
                padding: 1px;
            }

            QTreeWidget::item, QListWidget::item {
                border-radius: 3px;
                padding: 1px 3px;
            }

            QTreeWidget::item:selected, QListWidget::item:selected {
                background: #e0e7ff;
                color: #312e81;
            }

            QHeaderView::section {
                background: #eef2f7;
                color: #374151;
                border: none;
                border-bottom: 1px solid #d7deea;
                padding: 4px;
                font-weight: 700;
            }

            QWidget[panel="plot"] {
                background: #fafafa;
                border: none;
                border-radius: 0px;
            }

            QDockWidget {
                color: #1f2937;
                font-weight: 400;
            }

            QDockWidget::title {
                background: #e9eef7;
                font-weight: 400;
                border: 1px solid #d7deea;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 5px 8px;
                text-align: left;
            }

            QDockWidget > QWidget {
                background: #f8fafc;
                border: 1px solid #d7deea;
                border-top: none;
            }

            QProgressBar {
                background: #e8edf5;
                border: none;
                border-radius: 5px;
                min-height: 12px;
                text-align: center;
                color: #111827;
                font-weight: 400;
            }

            QProgressBar[lightText="true"] {
                color: #ffffff;
            }

            QProgressBar[lightText="false"] {
                color: #111827;
            }

            QProgressBar::chunk {
                background: #4f46e5;
                border-radius: 5px;
            }

            QStatusBar {
                background: #ffffff;
                border-top: 1px solid #dfe5ef;
                color: #667085;
            }

            QScrollBar:vertical {
                background: transparent;
                width: 9px;
                margin: 1px;
            }

            QScrollBar::handle:vertical {
                background: #c7cfdb;
                border-radius: 4px;
                min-height: 20px;
            }

            QScrollBar::handle:vertical:hover {
                background: #9faabd;
            }

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
                border: none;
                height: 0px;
            }

            QToolTip {
                background: #111827;
                color: #ffffff;
                border: 1px solid #374151;
                border-radius: 4px;
                padding: 3px;
            }
            """
        )

        # Keep unavailable fit values visually synchronized with labels such
        # as "None". The controller may update these fields after changing
        # the fit model, so refresh once the signal handlers have completed.
        for index in range(11):
            edit = getattr(self, f"fit{index}_edit", None)
            if edit is not None:
                edit.textChanged.connect(self._update_fit_unavailable_fields)

        self.fitModel_dropdown.currentIndexChanged.connect(
            lambda _index: QtCore.QTimer.singleShot(
                0, self._update_fit_unavailable_fields
            )
        )
        self._update_fit_unavailable_fields()

        # Keep progress text readable against the filled portion. Values above
        # 50% use white text; values at or below 50% use dark text.
        self.progressBar.valueChanged.connect(self._update_progress_text_color)
        self._update_progress_text_color(self.progressBar.value())

        # Conservative high-DPI overrides. The normal 96-DPI desktop appearance
        # stays essentially unchanged, while laptop/high-DPI screens get slightly
        # larger text and Matplotlib labels.
        self._apply_adaptive_font_overrides(MainWindow)
        self._scale_embedded_matplotlib_fonts()

        # Re-apply compact fixed heights after all style-sheet/font changes.
        # This is critical when Windows/Qt changes DPI mode after an external
        # monitor is attached.
        self._apply_dense_control_geometry(MainWindow)
        QtCore.QTimer.singleShot(
            0, lambda: self._apply_dense_control_geometry(MainWindow)
        )

        # The generated UI creates its fit controls out of display order.
        # Define a visual focus chain so Tab moves through each parameter's
        # checkbox and value field from top to bottom.
        fit_tab_widgets = [self.fitModel_dropdown]
        for index in range(11):
            fit_tab_widgets.extend((
                getattr(self, f"fit{index}_checkBox"),
                getattr(self, f"fit{index}_edit"),
            ))
        fit_tab_widgets.extend((
            self.fitweighted_checkBox,
            self.fitstart_spinBox,
            self.fitstop_spinBox,
            self.overwriteFit_button,
            self.newFit_button,
        ))
        for current_widget, next_widget in zip(
            fit_tab_widgets,
            fit_tab_widgets[1:],
        ):
            QtWidgets.QWidget.setTabOrder(current_widget, next_widget)

        # On small laptop screens the side docks become vertically cramped when
        # all panels are stacked above each other. In that situation, group the
        # side panels into tabbed dock stacks automatically. On large screens the
        # original side-by-side/stacked layout is preserved.
        self._auto_tabify_docks_for_small_screens(MainWindow)
        QtCore.QTimer.singleShot(0, lambda: self._auto_tabify_docks_for_small_screens(MainWindow))

        # Ensure all dock-panel titles use normal-weight text, including on
        # platform styles that do not fully honor the QDockWidget rule above.
        for dock in MainWindow.findChildren(QtWidgets.QDockWidget):
            dock_font = dock.font()
            dock_font.setBold(False)
            dock_font.setWeight(QtGui.QFont.Normal)
            dock.setFont(dock_font)

        # Re-polish widgets so dynamic properties are reflected immediately.
        for widget in MainWindow.findChildren(QtWidgets.QWidget):
            widget.style().unpolish(widget)
            widget.style().polish(widget)


    def _auto_tabify_docks_for_small_screens(self, MainWindow):
        """Use tabbed dock stacks automatically on compact displays.

        The generated UI places several dock widgets on the left and right. On
        a desktop monitor this is useful, but on a laptop screen the docks can
        be squeezed into very short panels. When the available screen area is
        small, tabifying the docks keeps each panel usable without changing any
        widget names or signal wiring.
        """
        screen = MainWindow.screen()
        if screen is None:
            app = QtWidgets.QApplication.instance()
            screen = app.primaryScreen() if app is not None else None
        if screen is None:
            return

        available = screen.availableGeometry()
        compact_screen = available.width() < 1450 or available.height() < 900
        if not compact_screen:
            return

        # Avoid doing this repeatedly when delayed layout passes run.
        if getattr(self, "_small_screen_docks_tabified", False):
            return
        self._small_screen_docks_tabified = True

        MainWindow.setDockOptions(
            QtWidgets.QMainWindow.AnimatedDocks
            | QtWidgets.QMainWindow.AllowNestedDocks
            | QtWidgets.QMainWindow.AllowTabbedDocks
        )

        # Right-side analysis panels: use one tab stack instead of three short
        # panels above each other. Raise Calculate correlations because this is
        # normally the first panel used in the workflow.
        MainWindow.tabifyDockWidget(self.dockWidget_3, self.dockWidget_4)
        MainWindow.tabifyDockWidget(self.dockWidget_3, self.dockWidget_5)
        self.dockWidget_3.raise_()

        # Keep the left-side image, notes, and progress panels visible one
        # above the other. Only the right-side analysis panels use tab stacks.

        # Make the tabs visible and compact enough for laptop use.
        for dock in (
            self.dockWidget, self.dockWidget_2, self.dockWidget_3,
            self.dockWidget_4, self.dockWidget_5, self.dockWidget_6,
        ):
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetFloatable
                | QtWidgets.QDockWidget.DockWidgetMovable
            )
            dock.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)


    def _apply_adaptive_font_overrides(self, MainWindow):
        """Apply small DPI-aware font corrections without hard-coding huge sizes."""
        font_pt = 9.0 * self._ui_scale
        title_pt = 11.0 * self._ui_scale
        subtitle_pt = 8.5 * self._ui_scale
        MainWindow.setStyleSheet(
            MainWindow.styleSheet()
            + f"""
            QWidget {{
                font-size: {font_pt:.2f}pt;
            }}
            QLabel[role="title"] {{
                font-size: {title_pt:.2f}pt;
            }}
            QLabel[role="subtitle"] {{
                font-size: {subtitle_pt:.2f}pt;
            }}
            QPushButton {{
                min-height: {self._button_height}px;
            }}
            QPushButton[role="navigation"], QPushButton[role="fileTab"] {{
                min-height: {self._top_button_height}px;
            }}
            QLineEdit, QSpinBox {{
                min-height: {self._field_height}px;
            }}
            QComboBox {{
                min-height: {self._combo_height}px;
                max-height: {self._combo_height + 1}px;
            }}
            """
        )

    def _scale_embedded_matplotlib_fonts(self):
        """Scale Matplotlib labels only modestly on high-DPI screens."""
        if _mpl is None:
            return

        scale = self._ui_scale
        mpl_values = {
            "font.size": 9.0 * scale,
            "axes.labelsize": 9.5 * scale,
            "axes.titlesize": 9.5 * scale,
            "xtick.labelsize": 8.5 * scale,
            "ytick.labelsize": 8.5 * scale,
            "legend.fontsize": 8.5 * scale,
            "figure.titlesize": 9.5 * scale,
            "axes.labelpad": 4.0 * scale,
            "xtick.major.pad": 3.0 * scale,
            "ytick.major.pad": 3.0 * scale,
        }
        _mpl.rcParams.update(mpl_values)

        for mpl_widget in (
            self.image_widget,
            self.fingerprint_widget,
            self.timetrace_widget,
            self.correlations_widget,
            self.difflaw_widget,
        ):
            canvas = getattr(mpl_widget, "canvas", None)
            figure = getattr(canvas, "figure", None)
            if figure is None:
                figure = getattr(mpl_widget, "figure", None)
            if figure is None:
                continue

            for axis in figure.get_axes():
                axis.title.set_fontsize(mpl_values["axes.titlesize"])
                axis.xaxis.label.set_fontsize(mpl_values["axes.labelsize"])
                axis.yaxis.label.set_fontsize(mpl_values["axes.labelsize"])
                axis.tick_params(
                    axis="both",
                    which="major",
                    labelsize=mpl_values["xtick.labelsize"],
                    pad=mpl_values["xtick.major.pad"],
                )
                axis.tick_params(
                    axis="both",
                    which="minor",
                    labelsize=max(7.0, 8.0 * scale),
                    pad=max(2.0, 2.5 * scale),
                )
                legend = axis.get_legend()
                if legend is not None:
                    for text in legend.get_texts():
                        text.set_fontsize(mpl_values["legend.fontsize"])
            if canvas is not None:
                canvas.draw_idle()

    def _update_progress_text_color(self, value):
        """Switch progress text color according to the normalized percentage."""
        minimum = self.progressBar.minimum()
        maximum = self.progressBar.maximum()

        if maximum > minimum:
            percentage = 100.0 * (value - minimum) / (maximum - minimum)
        else:
            percentage = 0.0

        use_light_text = percentage > 50.0
        if bool(self.progressBar.property("lightText")) != use_light_text:
            self.progressBar.setProperty("lightText", use_light_text)
            self.progressBar.style().unpolish(self.progressBar)
            self.progressBar.style().polish(self.progressBar)
            self.progressBar.update()

    def _update_fit_unavailable_fields(self, *_args):
        """Style unavailable values without muting real parameter names."""
        for index in range(11):
            checkbox = getattr(self, f"fit{index}_checkBox", None)
            edit = getattr(self, f"fit{index}_edit", None)
            if checkbox is None or edit is None:
                continue

            is_parameter = checkbox.text().strip().casefold() != "none"
            if bool(checkbox.property("fitParameter")) != is_parameter:
                checkbox.setProperty("fitParameter", is_parameter)
                checkbox.style().unpolish(checkbox)
                checkbox.style().polish(checkbox)
                checkbox.update()

            unavailable = (
                checkbox.text().strip().casefold() == "none"
                and edit.text().strip().casefold() == "nan"
            )

            if bool(edit.property("fitUnavailable")) != unavailable:
                edit.setProperty("fitUnavailable", unavailable)
                edit.style().unpolish(edit)
                edit.style().polish(edit)
                edit.update()


    def _apply_dense_control_geometry(self, MainWindow):
        """Force compact, DPI-stable geometry for dense form controls.

        The generated UI and Qt style engine can produce very different
        QComboBox size hints when the same laptop is used with an external
        monitor. This method intentionally fixes only the dense controls in
        the right-side docks. Larger main-window widgets keep their responsive
        behavior.
        """
        compact_combo_h = 19
        compact_field_h = 21
        compact_button_h = 22

        dense_combos = (
            self.fitModel_dropdown,
            self.corrs_dropdown,
            self.algorithm_dropdown,
            self.detector_dropdown,
            self.keepFixed_dropdown,
            self.calcDiameter_dropdown,
            self.showchunkscorr_dropdown,
            self.difflaw_dropdown,
        )

        for combo in dense_combos:
            combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            combo.setMinimumHeight(compact_combo_h)
            combo.setMaximumHeight(compact_combo_h)
            combo.setFixedHeight(compact_combo_h)
            combo.setStyleSheet(
                "QComboBox {"
                "  min-height: 0px;"
                f"  max-height: {compact_combo_h}px;"
                f"  height: {compact_combo_h}px;"
                "  padding: 0px 5px;"
                "}"
                "QComboBox::drop-down {"
                "  width: 16px;"
                "  border: none;"
                "}"
            )
            view = combo.view()
            if view is not None:
                popup_item_h = 22
                popup_rows = max(1, min(combo.count(), 18))
                combo.setMaxVisibleItems(max(8, min(25, combo.count())))
                view.setMinimumHeight(popup_rows * popup_item_h + 8)
                view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
                view.setItemDelegate(_CleanComboItemDelegate(view, minimum_height=popup_item_h))
                view.setStyleSheet(
                    "QAbstractItemView {"
                    "  background: #ffffff;"
                    "  border: 1px solid #cfd8e6;"
                    "  border-radius: 3px;"
                    "  padding: 1px;"
                    "  outline: none;"
                    "}"
                    "QAbstractItemView::item {"
                    "  min-height: 22px;"
                    "  padding: 1px 7px;"
                    "  border: none;"
                    "  border-radius: 3px;"
                    "}"
                    "QAbstractItemView::item:selected {"
                    "  background: #e8f5ee;"
                    "  color: #166534;"
                    "}"
                    "QAbstractItemView::indicator {"
                    "  width: 0px; height: 0px; image: none; border: none;"
                    "}"
                )

        dense_line_edits = [
            self.resolution_edit, self.chunkSize_edit,
            self.w0_edit, self.D_edit, self.T_edit, self.visc_edit, self.diameter_edit,
        ] + [getattr(self, f"fit{i}_edit") for i in range(11)]

        for edit in dense_line_edits:
            edit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            edit.setMinimumHeight(compact_field_h)
            edit.setMaximumHeight(compact_field_h)
            edit.setFixedHeight(compact_field_h)

        # A checkbox's size hint must not stretch an individual Fit row.
        for row in range(11):
            self.gridLayout_9.setRowMinimumHeight(row, compact_field_h)
            self.gridLayout_9.setRowStretch(row, 0)

        dense_spinboxes = (self.fitstart_spinBox, self.fitstop_spinBox)
        for spinbox in dense_spinboxes:
            spinbox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            spinbox.setMinimumHeight(compact_field_h)
            spinbox.setMaximumHeight(compact_field_h)
            spinbox.setFixedHeight(compact_field_h)
            spinbox.lineEdit().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            # Keep the number text optically centered in the compact spin box.
            # Some DPI/style combinations apply asymmetric bottom padding,
            # which makes the value appear too low.
            spinbox.setStyleSheet(
                "QSpinBox {"
                "  padding: 0px 6px 0px 6px;"
                f"  min-height: {compact_field_h}px;"
                f"  max-height: {compact_field_h}px;"
                "}"
            )

        dense_buttons = (
            self.addToJoblist_button,
            self.calcCorrCurrentFile_button,
            self.overwriteFit_button,
            self.newFit_button,
            self.updateDiffLaw_button,
            self.updateDiffLawDiameter_button,
        )
        for button in dense_buttons:
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            button.setMinimumHeight(compact_button_h)
            button.setMaximumHeight(compact_button_h)
            button.setFixedHeight(compact_button_h)

        fit_checks = tuple(getattr(self, f"fit{i}_checkBox") for i in range(11)) + (self.fitweighted_checkBox,)
        for checkbox in fit_checks:
            checkbox.setMinimumHeight(compact_field_h)
            checkbox.setMaximumHeight(compact_field_h)
            checkbox.setFixedHeight(compact_field_h)

        self.gridLayout_7.setVerticalSpacing(3)
        self.gridLayout_9.setVerticalSpacing(3)
        self.gridLayout_10.setVerticalSpacing(3)
        self.verticalLayout_4.setSpacing(3)
        self.verticalLayout_8.setSpacing(3)

        # The plot selectors sit above plots and should not stretch across the
        # whole plot width. Keep them compact: wide enough for the longest item
        # text plus the drop-down arrow, but not expanding with the available
        # horizontal space.
        compact_plot_selectors = (
            (self.showchunkscorr_dropdown, 150, 260),
            (self.difflaw_dropdown, 150, 235),
        )
        for combo, minimum_width, maximum_width in compact_plot_selectors:
            font_metrics = combo.fontMetrics()
            text_width = max(
                font_metrics.horizontalAdvance(combo.itemText(i))
                for i in range(combo.count())
            )
            combo_width = max(minimum_width, min(maximum_width, text_width + 42))
            combo.setSizePolicy(
                QtWidgets.QSizePolicy.Fixed,
                QtWidgets.QSizePolicy.Fixed,
            )
            combo.setMinimumWidth(combo_width)
            combo.setMaximumWidth(combo_width)
            combo.setFixedWidth(combo_width)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "BrightEyes FFS Analysis"))
        self.prevFCSfile_button.setText("")
        self.FCSfile0_button.setText(_translate("MainWindow", "New file"))
        self.FCSfile1_button.setText(_translate("MainWindow", "File 2"))
        self.FCSfile2_button.setText(_translate("MainWindow", "File 3"))
        self.FCSfile3_button.setText(_translate("MainWindow", "File 4"))
        self.FCSfile4_button.setText(_translate("MainWindow", "File 5"))
        self.nextFCSfile_button.setText("")
        self.FCSFileName_label.setText(_translate("MainWindow", "My FCS File"))
        self.FCSFolderName_label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#b1b1b1;\">C:\\Users\\Me\\My FCS Folder</span></p></body></html>"))
        self.label_edit.setText(_translate("MainWindow", "Label"))
        self.ycoord_edit.setText(_translate("MainWindow", "Y"))
        self.xcoord_edit.setText(_translate("MainWindow", "X"))
        self.saveLabel_button.setText(_translate("MainWindow", "ok"))
        self.correlations_treeWidget.headerItem().setText(0, _translate("MainWindow", "Correlation analysis"))
        __sortingEnabled = self.showElements_widget.isSortingEnabled()
        self.showElements_widget.setSortingEnabled(False)
        item = self.showElements_widget.item(0)
        item.setText(_translate("MainWindow", "Central"))
        item = self.showElements_widget.item(1)
        item.setText(_translate("MainWindow", "Spot-variation"))
        item = self.showElements_widget.item(2)
        item.setText(_translate("MainWindow", "All individually"))
        self.showElements_widget.setSortingEnabled(__sortingEnabled)
        self.chunkOn_checkBox.setText(_translate("MainWindow", "Chunk on"))
        self.showchunkscorr_dropdown.setItemText(0, _translate("MainWindow", "Show current chunk"))
        self.showchunkscorr_dropdown.setItemText(1, _translate("MainWindow", "Show average all active chunks"))
        self.difflaw_dropdown.setItemText(0, _translate("MainWindow", "Diffusion law"))
        self.difflaw_dropdown.setItemText(1, _translate("MainWindow", "Particle number"))
        self.difflaw_dropdown.setItemText(2, _translate("MainWindow", "Number and brightness"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuCalculate.setTitle(_translate("MainWindow", "Calculate"))
        self.menuRemove.setTitle(_translate("MainWindow", "Remove"))
        self.menuTools.setTitle(_translate("MainWindow", "Tools"))
        self.dockWidget.setWindowTitle(_translate("MainWindow", "Image"))
        self.prevImage_button.setText("")
        self.imageName_button.setText(_translate("MainWindow", "PushButton"))
        self.nextImage_button.setText("")
        self.imageInfo_label.setText(_translate("MainWindow", "Image info"))
        self.dockWidget_6.setWindowTitle(_translate("MainWindow", "Notes"))
        self.dockWidget_2.setWindowTitle(_translate("MainWindow", "Status"))
        self.progressBar_label.setText(_translate("MainWindow", "Done."))
        self.dockWidget_3.setWindowTitle(_translate("MainWindow", "Calculate correlations"))
        self.corrs_dropdown.setItemText(0, _translate("MainWindow", "Spot-variation FCS"))
        self.corrs_dropdown.setItemText(1, _translate("MainWindow", "Cross-correlation spectroscopy"))
        self.corrs_dropdown.setItemText(2, _translate("MainWindow", "Pair-correlation FCS"))
        self.corrs_dropdown.setItemText(3, _translate("MainWindow", "All autocorrelations"))
        self.corrs_dropdown.setItemText(4, _translate("MainWindow", "STICS with iMSD analysis"))
        self.calcCorrCurrentFile_button.setText(_translate("MainWindow", "Calculate now"))
        self.addToJoblist_button.setText(_translate("MainWindow", "Add to joblist"))
        self.precision_label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:8pt;\">Precision</span></p></body></html>"))
        self.resolution_edit.setText(_translate("MainWindow", "10"))
        self.chunkSize_label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:8pt;\">Chunk size (s)</span></p></body></html>"))
        self.chunkSize_edit.setText(_translate("MainWindow", "10"))
        self.algorithm_dropdown.setItemText(0, _translate("MainWindow", "Multipletau"))
        self.algorithm_dropdown.setItemText(1, _translate("MainWindow", "Wiener-Khinchin"))
        self.algorithm_dropdown.setItemText(2, _translate("MainWindow", "TT2Corr"))
        self.algorithm_dropdown.setItemText(3, _translate("MainWindow", "PCH"))
        self.algorithm_label.setText(_translate("MainWindow", "Algorithm"))
        self.label.setText(_translate("MainWindow", "Detector"))
        self.detector_dropdown.setItemText(0, _translate("MainWindow", "Square 5x5"))
        self.detector_dropdown.setItemText(1, _translate("MainWindow", "PDA-23"))
        self.detector_dropdown.setItemText(2, _translate("MainWindow", "Airyscan 32"))
        self.detector_dropdown.setItemText(3, _translate("MainWindow", "PRISM 7x7"))
        self.dockWidget_4.setWindowTitle(_translate("MainWindow", "Fit analysis"))
        self.fitModel_dropdown.setItemText(0, _translate("MainWindow", "Free diffusion 1 component"))
        self.fitModel_dropdown.setItemText(1, _translate("MainWindow", "Anomalous diffusion 1 component"))
        self.fitModel_dropdown.setItemText(2, _translate("MainWindow", "Free diffusion circular scanning"))
        self.fitModel_dropdown.setItemText(3, _translate("MainWindow", "Free diffusion 2 components"))
        self.fitModel_dropdown.setItemText(4, _translate("MainWindow", "2 components with afterpulsing"))
        self.fit2_edit.setText(_translate("MainWindow", "3,3,3"))
        self.fit8_edit.setText(_translate("MainWindow", "NaN"))
        self.fitstop_label.setText(_translate("MainWindow", "Stop"))
        self.fitstart_label.setText(_translate("MainWindow", "Start"))
        self.fit5_checkBox.setText(_translate("MainWindow", "None"))
        self.fit0_edit.setText(_translate("MainWindow", "1,1,1"))
        self.fit7_edit.setText(_translate("MainWindow", "NaN"))
        self.overwriteFit_button.setText(_translate("MainWindow", "Overwrite fit"))
        self.fit1_edit.setText(_translate("MainWindow", "1,1,1"))
        self.fit7_checkBox.setText(_translate("MainWindow", "None"))
        self.fit5_edit.setText(_translate("MainWindow", "NaN"))
        self.newFit_button.setText(_translate("MainWindow", "New fit"))
        self.fit6_checkBox.setText(_translate("MainWindow", "None"))
        self.fit8_checkBox.setText(_translate("MainWindow", "None"))
        self.fit4_edit.setText(_translate("MainWindow", "NaN"))
        self.fit3_edit.setText(_translate("MainWindow", "0,0,0"))
        self.fit4_checkBox.setText(_translate("MainWindow", "None"))
        self.fit6_edit.setText(_translate("MainWindow", "NaN"))
        self.fit9_edit.setText(_translate("MainWindow", "NaN"))
        self.fit9_checkBox.setText(_translate("MainWindow", "None"))
        self.fit10_checkBox.setText(_translate("MainWindow", "None"))
        self.fit10_edit.setText(_translate("MainWindow", "NaN"))
        self.fit1_checkBox.setText(_translate("MainWindow", "Tau (ms)"))
        self.fit0_checkBox.setText(_translate("MainWindow", "N"))
        self.fit3_checkBox.setText(_translate("MainWindow", "Offset"))
        self.fit2_checkBox.setText(_translate("MainWindow", "Shape parameter"))
        self.fitweighted_checkBox.setText(_translate("MainWindow", "Weighted fit"))
        self.dockWidget_5.setWindowTitle(_translate("MainWindow", "Diffusion analysis"))
        self.visc_edit.setText(_translate("MainWindow", "0.001"))
        self.viscosity_label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:8pt;\">Viscosity (Pa.s)</span></p></body></html>"))
        self.w0_label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:8pt;\">Beam waist (nm)</span></p></body></html>"))
        self.D_edit.setText(_translate("MainWindow", "10.0"))
        self.D_label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:8pt;\">D (µm^2/s)</span></p></body></html>"))
        self.w0_edit.setText(_translate("MainWindow", "250, 300, 350"))
        self.T_edit.setText(_translate("MainWindow", "293"))
        self.temperature_label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:8pt;\">Temperature (K)</span></p></body></html>"))
        self.keepFixed_dropdown.setItemText(0, _translate("MainWindow", "Keep w0 fixed"))
        self.keepFixed_dropdown.setItemText(1, _translate("MainWindow", "Keep D fixed"))
        self.updateDiffLaw_button.setText(_translate("MainWindow", "Update"))
        self.calcDiameter_dropdown.setItemText(0, _translate("MainWindow", "Calculate diameter"))
        self.calcDiameter_dropdown.setItemText(1, _translate("MainWindow", "Calculate viscosity"))
        self.calcDiameter_dropdown.setItemText(2, _translate("MainWindow", "Calculate temperature"))
        self.diameter_label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:8pt;\">Diameter (nm)</span></p></body></html>"))
        self.diameter_edit.setText(_translate("MainWindow", "10, 10, 10"))
        self.updateDiffLawDiameter_button.setText(_translate("MainWindow", "Update"))
        self.actionOpen_image.setText(_translate("MainWindow", "Open new image"))
        self.actionOpen_image.setShortcut(_translate("MainWindow", "Ctrl+I"))
        self.actionChange_image.setText(_translate("MainWindow", "Change image"))
        self.actionOpen_session.setText(_translate("MainWindow", "Open session"))
        self.actionOpen_session.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionSave_session.setText(_translate("MainWindow", "Save session"))
        self.actionSave_session.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionNew_session.setText(_translate("MainWindow", "New session"))
        self.actionNew_session.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.actionCorrelation.setText(_translate("MainWindow", "Correlation"))
        self.actionAllCorrelationsCurrentFile.setText(_translate("MainWindow", "All correlations in joblist"))
        self.actionCurrent_fit.setText(_translate("MainWindow", "Current fit"))
        self.actionCurrent_fit.setShortcut(_translate("MainWindow", "Del"))
        self.actionCurrent_correlation.setText(_translate("MainWindow", "Current correlation"))
        self.det_Genoa_Instrument_5x5.setText(_translate("MainWindow", "Genoa Instruments 5x5"))
        self.det_Genoa_Instruments_7x7.setText(_translate("MainWindow", "Genoa Instruments PRISM 7x7"))
        self.det_Nikon_NSPARC.setText(_translate("MainWindow", "Nikon NSPARC"))
        self.det_PI_Imaging_23.setText(_translate("MainWindow", "PI Imaging SPAD23"))
        self.actionSave_session_as.setText(_translate("MainWindow", "Save session as..."))
        self.actionSave_session_as.setShortcut(_translate("MainWindow", "Ctrl+Shift+S"))
        self.actionCurrent_FFS_file.setText(_translate("MainWindow", "Current FFS file"))
        self.actionCopy_current_fit_as_data.setText(_translate("MainWindow", "Copy current fit as data"))
        self.actionCreate_Jupyter_Notebook.setText(_translate("MainWindow", "Create Jupyter Notebook"))
        self.actionCopy_current_correlation.setText(_translate("MainWindow", "Copy current correlation"))
        self.actionCopy_current_correlation.setShortcut(_translate("MainWindow", "Ctrl+C"))
        self.actionFilter_out_bad_chunks.setText(_translate("MainWindow", "Filter out bad chunks"))
        self.actionFilter_out_bad_chunks.setShortcut(_translate("MainWindow", "Ctrl+F"))
        self.actionCurrent_Image.setText(_translate("MainWindow", "Current Image"))
        self.actionPlot_in_Jupyter_Notebook.setText(_translate("MainWindow", "Plot in Jupyter Notebook"))
        self.actionExport_results_to_Excel.setText(_translate("MainWindow", "Export results to Excel"))
        self.actionExport_current_correlation_to_Excel.setText(_translate("MainWindow", "Export correlation to Excel"))
from mplwidget import MplWidget
from mplwidgetcorrplot import MplWidgetCorrPlot
from mplwidgetdifflawplot import MplWidgetDiffLawPlot
from mplwidgetfingerprint import MplWidgetFingerPrint
from mplwidgetlineplot import MplWidgetLinePlot