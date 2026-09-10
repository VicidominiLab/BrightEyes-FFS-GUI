from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMenu, QAction
from PyQt5.QtCore import QSize, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib import gridspec


class CustomNavigationToolbar(NavigationToolbar2QT):
    toolitems = [
        ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        ('Save', 'Save the figure', 'filesave', 'save_figure'),
    ]

    def set_message(self, s):
        pass


class MplWidgetCorrPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        my_dpi = 120
        figure = Figure(
            figsize=(350 / my_dpi, 350 / my_dpi),
            dpi=my_dpi,
        )
        figure.set_facecolor("#FAFAFA")

        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

        self.canvas = FigureCanvas(figure)
        self._curve_groups = {}
        self._fit_groups = {}
        self._fit_residuals_groups = {}
        self._lock_axes_range = False

        self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._show_curve_menu)

        self.toolbar = CustomNavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #FAFAFA;")
        self.toolbar.setIconSize(QSize(12, 12))

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        vertical_layout.addWidget(self.toolbar)
        self.setLayout(vertical_layout)

        self.canvas.axes = self.canvas.figure.add_subplot(gs[0])
        self.canvas.axes2 = self.canvas.figure.add_subplot(gs[1])

        self.canvas.figure.subplots_adjust(
            bottom=0.15,
            top=0.95,
            left=0.2,
            right=0.95,
            hspace=0.4,
        )

    def empty_curve_register(self):
        curve_groups = self._curve_groups
        fit_groups = self._fit_groups
        fit_residuals_groups = self._fit_residuals_groups

        self._curve_groups = {}
        self._fit_groups = {}
        self._fit_residuals_groups = {}

        return curve_groups, fit_groups, fit_residuals_groups

    def register_curve(self, name, *artists, group_type="curve"):
        """Register a data curve or its corresponding fit.

        Curves and fits that should toggle together must have the same name.
        """
        if group_type == "curve":
            self._curve_groups[name] = artists
        elif group_type == "fit":
            self._fit_groups[name] = artists
        elif group_type == "residuals":
            self._fit_residuals_groups[name] = artists
        else:
            raise ValueError("group_type must be 'curve' or 'fit'")

    def _show_curve_menu(self, position):
        menu = QMenu(self)

        if self._lock_axes_range:
            action = QAction(f"Unlock axes range", menu)
        else:
            action = QAction(f"Lock axes range", menu)

        action.triggered.connect(lambda: self._update_lock_axes_range())
        menu.addAction(action)

        menu.addSeparator()

        for name, artists in self._curve_groups.items():
            is_visible = all(artist.get_visible() for artist in artists)
            status = "" if is_visible else " [Off]"

            action = QAction(f"{name}{status}", menu)
            action.setCheckable(True)
            action.setChecked(is_visible)

            action.triggered.connect(
                lambda checked, curve_name=name, popup=menu:
                    self._set_curve_and_fit_visible(
                        curve_name, checked, popup
                    )
            )
            menu.addAction(action)

        if not self._curve_groups:
            empty_action = QAction("No curves available", menu)
            empty_action.setEnabled(False)
            menu.addAction(empty_action)

        menu.exec_(self.canvas.mapToGlobal(position))

    def _set_curve_and_fit_visible(self, name, visible, menu):
        """Toggle a curve and the fit registered with the same name."""
        for artist in self._curve_groups[name]:
            artist.set_visible(visible)

        for artist in self._fit_groups.get(name, ()):
            artist.set_visible(visible)

        for artist in self._fit_residuals_groups.get(name, ()):
            artist.set_visible(visible)

        self.canvas.draw_idle()
        menu.close()

    def _update_lock_axes_range(self):
        self._lock_axes_range = not self._lock_axes_range

    def restore_curve_visibility(self, curve_groups_old, fit_groups_old, fit_residuals_groups_old):
        """Restore visibility only when all registered names match exactly."""
        group_pairs = ((curve_groups_old, self._curve_groups), (fit_groups_old, self._fit_groups), (fit_residuals_groups_old, self._fit_residuals_groups))

        # Do nothing unless every old/new register is an exact name-for-name match.
        for old_groups, new_groups in group_pairs:
            if (len(old_groups) != len(new_groups) or set(old_groups) != set(new_groups)):
                return

        # The dictionaries match: copy each group's old visible/hidden state.
        for old_groups, new_groups in group_pairs:
            for name, old_artists in old_groups.items():
                was_visible = all(artist.get_visible() for artist in old_artists)

                for artist in new_groups[name]:
                    artist.set_visible(was_visible)

        self.canvas.draw_idle()