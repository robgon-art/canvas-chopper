import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QStatusBar, QFileDialog, QAction

class STLViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stratogrpah Viewer")
        self.setGeometry(100, 100, 800, 800)

        # Create the central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create the VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor()
        self.vtk_widget.setParent(central_widget)
        self.vtk_widget.setGeometry(0, 0, 800, 800)

        # Create the renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(vtk.vtkNamedColors().GetColor3d("DarkGray"))
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        # Create the interactor style
        self.interactor_style = vtk.vtkInteractorStyleTrackballActor()
        self.vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(self.interactor_style)

        # Create the status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Create the file menu
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        # Create the Load STL action
        load_stl_action = QAction("Load STL", self)
        load_stl_action.triggered.connect(self.load_stl)
        file_menu.addAction(load_stl_action)

        # Show the window
        self.show()

    def load_stl(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load STL File", "", "STL Files (*.stl)")
        if file_name:
            try:
                reader = vtk.vtkSTLReader()
                reader.SetFileName(file_name)
                reader.Update()

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(reader.GetOutputPort())

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)

                self.renderer.AddActor(actor)
                self.vtk_widget.GetRenderWindow().Render()

                self.status_bar.showMessage("File loaded successfully.")
            except Exception as e:
                self.status_bar.showMessage(f"Error loading STL file: {e}")

if __name__ == "__main__":
    app = QApplication([])
    viewer = STLViewer()
    app.exec_()
