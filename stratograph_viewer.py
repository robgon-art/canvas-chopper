import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QStatusBar, QFileDialog, QAction
import json
import os
from PyQt5.QtCore import Qt

# Function to read the file and return the layer data
def extract_layers(file_path):
    # Read the contents of the file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Helper function to convert hex color to RGB (normalized to 0-1 range)
    def hex_to_rgb_normalized(hex_color):
        hex_color = hex_color.lstrip('#')
        return [round(int(hex_color[i:i+2], 16) / 255, 2) for i in (0, 2, 4)]

    # Read the base_layer_height and layer_height values
    base_layer_height = data['base_layer_height']
    layer_height = data['layer_height']

    # Extract filament information in reverse order and calculate Depth
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
            'Depth': round(depth, 2)  # Round the depth for clarity
        })

    return layers

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

def export_layer_as_stl(layer_data, file_name):
    # Create an STL writer
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(layer_data)
    writer.Write()

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


class STLViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stratograph Viewer")
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
        self.renderer.SetBackground(vtk.vtkNamedColors().GetColor3d("DimGray"))
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

        # Add keyboard shortcuts for Blender-style navigation
        self.vtk_widget.GetRenderWindow().GetInteractor().AddObserver("KeyPressEvent", self.on_key_press)

        # Show the window
        self.show()

    def load_stl(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load STL File", "", "STL Files (*.stl)")
        if file_name:
            try:
                # Load the STL file
                reader = vtk.vtkSTLReader()
                reader.SetFileName(file_name)
                reader.Update()

                model_data = reader.GetOutput()

                # Find the corresponding hfp file
                stl_base_name = os.path.basename(file_name)
                if "_Front_" in stl_base_name:
                    prefix = stl_base_name.split("_Front_")[0]
                    hfp_file_name = f"{prefix}.hfp"
                    hfp_file_path = os.path.join(os.path.dirname(file_name), hfp_file_name)

                    if os.path.exists(hfp_file_path):
                        # Set the status bar message and change to a wait cursor
                        self.status_bar.showMessage("Loading the STL and HFP files.")
                        self.repaint()
                        self.setCursor(Qt.WaitCursor)
                        QApplication.processEvents()

                        layers = extract_layers(hfp_file_path)
                        print(f"Loaded hfp file: {hfp_file_name}")
                        for i, layer in enumerate(layers):
                            print(layer)

                            # Divide the model into layers and apply materials
                            assembly = divide_model_into_layers(model_data, layers)

                        # Add the assembly to the renderer instead of individual actors
                        self.renderer.AddActor(assembly)
                        self.renderer.ResetCamera()
                        self.vtk_widget.GetRenderWindow().Render()

                        # Reset the cursor and show a completion message
                        self.setCursor(Qt.ArrowCursor)
                        self.status_bar.showMessage("STL and HFP files loaded successfully.")
                    else:
                        print(f"No corresponding hfp file found: {hfp_file_name}")

            except Exception as e:
                self.status_bar.showMessage(f"Error loading STL file: {e}")

    def on_key_press(self, obj, event):
        key = obj.GetKeySym()
        camera = self.renderer.GetActiveCamera()
        
        # Move the camera around the object while maintaining focus
        if key == 'Left':  # Rotate view left
            camera.Azimuth(10)
        elif key == 'Right':  # Rotate view right
            camera.Azimuth(-10)
        elif key == 'Up':  # Rotate view up
            camera.Elevation(10)
        elif key == 'Down':  # Rotate view down
            camera.Elevation(-10)
        elif key == 'plus':  # Zoom in
            camera.Zoom(1.1)
        elif key == 'minus':  # Zoom out
            camera.Zoom(0.9)
        elif key == 'End':  # Reset azimuth and elevation to default orientation
            camera.SetPosition(0, 0, 493)  # Set default position (camera is 493 units away from the object)
            camera.SetFocalPoint(0, 0, 1.16) # Set focal point to the center
            camera.SetViewUp(0, 1, 0)  # Set the view-up vector (pointing up the Y axis)

        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

if __name__ == "__main__":
    app = QApplication([])
    viewer = STLViewer()
    app.exec_()
