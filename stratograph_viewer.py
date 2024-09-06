import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QStatusBar, QFileDialog, QAction
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
import json
import os

# Function to slice the model at a specific depth
def slice_model_at_depth(input_data, depth):
    # Create a plane to slice the model at the given depth
    plane = vtk.vtkPlane()
    plane.SetOrigin(0, 0, depth)  # Set the origin of the plane at the depth
    plane.SetNormal(0, 0, -1)     # Set the plane normal to point in the negative Z direction

    # Use vtkClipPolyData to clip the model below the plane
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(input_data)
    clipper.SetClipFunction(plane)
    clipper.Update()

    return clipper.GetOutput()

# Slicing the model at specific depths (lower and upper)
def slice_model_at_depth_range(input_data, lower_limit, upper_limit):
    # Clip the model below the upper limit
    plane_upper = vtk.vtkPlane()
    plane_upper.SetOrigin(0, 0, upper_limit)
    plane_upper.SetNormal(0, 0, -1)

    clip_upper = vtk.vtkClipPolyData()
    clip_upper.SetInputData(input_data)
    clip_upper.SetClipFunction(plane_upper)
    clip_upper.Update()

    # Clip the model above the lower limit
    plane_lower = vtk.vtkPlane()
    plane_lower.SetOrigin(0, 0, lower_limit)
    plane_lower.SetNormal(0, 0, 1)

    clip_lower = vtk.vtkClipPolyData()
    clip_lower.SetInputData(clip_upper.GetOutput())  # Clip the already clipped data
    clip_lower.SetClipFunction(plane_lower)
    clip_lower.Update()

    return clip_lower.GetOutput()

# Function to read the file and return the layer data
def extract_layers(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    def hex_to_rgb_normalized(hex_color):
        hex_color = hex_color.lstrip('#')
        return [round(int(hex_color[i:i+2], 16) / 255, 2) for i in (0, 2, 4)]

    base_layer_height = data['base_layer_height']
    layer_height = data['layer_height']

    layers = []
    for index, filament in enumerate(reversed(data['filament_set'])):
        rgb_normalized = hex_to_rgb_normalized(filament['Color'])
        slider_value = data['slider_values'][index]
        depth = base_layer_height + slider_value * layer_height
        layers.append({
            'Layer': index,
            'Brand': filament['Brand'],
            'Name': filament['Name'],
            'Color (RGB)': rgb_normalized,
            'Transmissivity': filament['Transmissivity'],
            'Type': filament['Type'],
            'Slider': slider_value,
            'Depth': round(depth, 2)
        })

    return layers

# Grouping the actors together using vtkAssembly
def divide_model_into_layers(input_data, layers):
    assembly = vtk.vtkAssembly()  # Create a vtkAssembly to group actors

    for i, layer in enumerate(layers):
        current_depth = layer['Depth']

        # Handle slicing logic based on whether it's the bottom, inner, or top layer
        if i == 0:
            # Bottom layer: Only slice at the upper limit
            layer_data = slice_model_at_depth(input_data, current_depth)
        elif i == len(layers) - 1:
            # Top layer: Only slice at the lower limit
            layer_data = slice_model_at_depth_range(input_data, previous_depth, current_depth)
        else:
            # Inner layers: Slice at both lower and upper limits
            layer_data = slice_model_at_depth_range(input_data, previous_depth, current_depth)

        # Create a mapper and actor for this layer
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(layer_data)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Set the material color for this layer using the RGB values
        rgb_color = layer['Color (RGB)']
        actor.GetProperty().SetColor(rgb_color)

        # Add the actor to the assembly
        assembly.AddPart(actor)

        # Update previous_depth for the next iteration
        previous_depth = current_depth

    return assembly

class FileLoaderWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(Exception)
    status_update = pyqtSignal(str)

    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name

    def run(self):
        try:
            self.status_update.emit("Loading STL file...")
            reader = vtk.vtkSTLReader()
            reader.SetFileName(self.file_name)
            reader.Update()

            model_data = reader.GetOutput()
            stl_base_name = os.path.basename(self.file_name)
            prefix = stl_base_name.split("_Front_")[0]
            hfp_file_name = f"{prefix}.hfp"
            hfp_file_path = os.path.join(os.path.dirname(self.file_name), hfp_file_name)

            if os.path.exists(hfp_file_path):
                layers = extract_layers(hfp_file_path)
                assembly = divide_model_into_layers(model_data, layers)
                self.status_update.emit("Rendering model...")
                self.finished.emit(assembly)
            else:
                self.error.emit(RuntimeError(f"No corresponding HFP file found: {hfp_file_name}"))
        except Exception as e:
            self.error.emit(e)

class STLViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stratograph Viewer")
        self.setGeometry(100, 100, 800, 800)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.vtk_widget = QVTKRenderWindowInteractor()
        self.vtk_widget.setParent(central_widget)
        self.vtk_widget.setGeometry(0, 0, 800, 800)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(vtk.vtkNamedColors().GetColor3d("DimGray"))
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor_style = vtk.vtkInteractorStyleTrackballActor()
        self.vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(self.interactor_style)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        load_stl_action = QAction("Load STL", self)
        load_stl_action.triggered.connect(self.load_stl)
        file_menu.addAction(load_stl_action)
        self.vtk_widget.GetRenderWindow().GetInteractor().AddObserver("KeyPressEvent", self.on_key_press)
        self.show()

    def load_stl(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load STL File", "", "STL Files (*.stl)")
        if file_name:
            self.worker = FileLoaderWorker(file_name)
            self.thread = QThread()
            self.worker.moveToThread(self.thread)
            self.worker.finished.connect(self.handle_finished)
            self.worker.error.connect(self.handle_error)
            self.worker.status_update.connect(self.status_bar.showMessage)
            QApplication.setOverrideCursor(Qt.WaitCursor)  # Show the watch cursor
            self.thread.started.connect(self.worker.run)
            self.thread.start()

    def handle_finished(self, assembly):
        self.renderer.AddActor(assembly)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        QApplication.restoreOverrideCursor()  # Restore the default cursor
        self.status_bar.showMessage("STL and HFP files loaded successfully.")
        self.thread.quit()
        self.thread.wait()

    def handle_error(self, e):
        QApplication.restoreOverrideCursor()  # Restore the default cursor
        self.status_bar.showMessage(f"Error loading STL file: {str(e)}")
        self.thread.quit()
        self.thread.wait()

    def on_key_press(self, obj, event):
        key = obj.GetKeySym()
        camera = self.renderer.GetActiveCamera()
        if key == 'Left':
            camera.Azimuth(10)
        elif key == 'Right':
            camera.Azimuth(-10)
        elif key == 'Up':
            camera.Elevation(10)
        elif key == 'Down':
            camera.Elevation(-10)
        elif key == 'plus':
            camera.Zoom(1.1)
        elif key == 'minus':
            camera.Zoom(0.9)
        elif key == 'End':
            camera.SetPosition(0, 0, 493)
            camera.SetFocalPoint(0, 0, 1.16)
            camera.SetViewUp(0, 1, 0)
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

if __name__ == "__main__":
    app = QApplication([])
    viewer = STLViewer()
    app.exec_()
