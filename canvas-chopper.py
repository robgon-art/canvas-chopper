import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageEnhance, ImageDraw
import onnxruntime as ort
import numpy as np
from copy import deepcopy
import threading
import colorsys

# Global variables to store the full-sized image, embeddings, segment masks, and scaling factor
full_size_image = None
resized_image = None
embeddings = None
segment_masks = []  # List to store each individual segment mask
scaling_factor = 1.0
phi = 0.61803398875  # Golden ratio conjugate
selection_points = []

# Global variable to track the index of the last active segment mask
current_index = -1

# Default DPI value
default_dpi = 60

# Function to calculate a unique color based on the golden ratio given an index
def calculate_color(index, saturation=1.0, value=1.0):
    hue = (index * phi * 360) % 360
    r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)

# Function to load and display the image
def load_image():
    global full_size_image, resized_image, embeddings, segment_masks, scaling_factor, current_index

    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )
    
    if file_path:
        # Start a new thread to process the image
        root.config(cursor="watch")  # Set cursor to watch
        threading.Thread(target=process_image, args=(file_path,)).start()

def process_image(file_path):
    global full_size_image, resized_image, embeddings, segment_masks, scaling_factor, current_index

    # Update the instruction label to show the model is running
    instruction_label.config(text="Running the Segment Anything model.")
    root.update_idletasks()  # Ensure the text update is displayed immediately

    # Open the image and store the full-sized image
    full_size_image = Image.open(file_path).convert("RGB")
    
    # Calculate the aspect ratio and resize for display only
    orig_width, orig_height = full_size_image.size
    new_width = 512
    new_height = int((new_width / orig_width) * orig_height)
    scaling_factor = orig_width / new_width
    
    # Resize the image for display only
    resized_image = full_size_image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    # Clear the segment masks list and reset the current index for new image
    segment_masks.clear()
    current_index = -1

    # Update the DPI-dependent size display
    update_image_size_display()

    # Prepare input tensor from the original full-size image for encoder
    input_tensor = np.array(full_size_image)
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    input_tensor = (input_tensor - mean) / std
    input_tensor = input_tensor.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
    
    # Pad input tensor to 1024x1024
    pad_height = 1024 - input_tensor.shape[2]
    pad_width = 1024 - input_tensor.shape[3]
    input_tensor = np.pad(input_tensor, ((0, 0), (0, 0), (0, pad_height), (0, pad_width)))

    # Run the image encoder to get embeddings
    encoder = ort.InferenceSession("sam_encoder.onnx")
    outputs = encoder.run(None, {"images": input_tensor})
    embeddings = outputs[0]

    # Convert the image to a Tkinter-compatible format and display it
    photo = ImageTk.PhotoImage(resized_image)
    image_label.config(image=photo)
    image_label.image = photo  # Keep a reference to avoid garbage collection

    # Update the instruction label after the image is loaded
    instruction_label.config(text="Click in the image to define segments.")

    # Reset the cursor back to the default arrow
    root.config(cursor="arrow")

    # Clear the segment list in the UI
    clear_segment_list()

# Function to handle mouse clicks on the image
def on_image_click(event):
    global embeddings, resized_image, segment_masks, scaling_factor, current_index, selection_points

    if embeddings is not None:
        # Calculate the clicked coordinates in the original image size
        input_point = np.array([[int(event.x * scaling_factor), int(event.y * scaling_factor)]])
        input_label = np.array([1])

        onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, np.array([-1])])[None, :].astype(np.float32)

        coords = deepcopy(onnx_coord).astype(float)
        coords[..., 0] = coords[..., 0] * (1024 / full_size_image.width)
        coords[..., 1] = coords[..., 1] * (1024 / full_size_image.height)
        onnx_coord = coords.astype("float32")

        # Prepare inputs for the decoder
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        # Run the decoder to generate the mask
        decoder = ort.InferenceSession("sam_decoder.onnx")
        masks_output, _, _ = decoder.run(None, {
            "image_embeddings": embeddings,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array([full_size_image.height, full_size_image.width], dtype=np.float32)
        })

        # Post-process the mask
        mask = masks_output[0][0]
        mask = (mask > 0).astype('uint8') * 255

        # If a new mask is created, discard any masks after the current index
        segment_masks = segment_masks[:current_index + 1]
        selection_points = selection_points[:current_index + 1]

        # Store the individual segment mask and point in the list
        segment_masks.append(mask)
        selection_points.append((input_point[0], (event.x, event.y)))
        current_index += 1

        # Overlay the mask and draw the points on the image
        overlay_image()

        # Update the segment list in the UI
        update_segment_list()

# Function to overlay the mask on the image
def overlay_image():
    global resized_image, segment_masks, current_index, selection_points

    # Start with a transparent overlay
    combined_overlay = Image.new("RGBA", resized_image.size, (255, 0, 0, 0))

    # Apply each mask up to the current index with its corresponding unique color
    for index in range(current_index + 1):
        mask_resized = Image.fromarray(segment_masks[index]).resize(resized_image.size, Image.Resampling.NEAREST)
        color = calculate_color(index)
        colored_overlay = Image.new("RGBA", resized_image.size, color + (128,))
        combined_overlay.paste(colored_overlay, mask=mask_resized)

    # Combine the original image with the combined overlay
    combined_image = Image.alpha_composite(resized_image.convert("RGBA"), combined_overlay)

    # Draw circles and numbers for each selection point
    draw = ImageDraw.Draw(combined_image)
    for index, (point, screen_coord) in enumerate(selection_points[:current_index + 1]):
        color = calculate_color(index, saturation=0.5)
        circle_radius = 10
        x, y = screen_coord
        draw.ellipse((x - circle_radius, y - circle_radius, x + circle_radius, y + circle_radius), fill=color)
        text = str(index + 1)
        text_width, text_height = draw.textsize(text)
        text_x = x - text_width / 2 + 1
        text_y = y - text_height / 2 + 1
        draw.text((text_x, text_y), text, fill="black")

    # Convert to Tkinter-compatible format and display it
    photo = ImageTk.PhotoImage(combined_image.convert("RGB"))
    image_label.config(image=photo)
    image_label.image = photo  # Keep a reference to avoid garbage collection

# Function to handle the undo operation (Ctrl-Z)
def undo(event=None):
    global current_index
    if current_index >= 0:
        current_index -= 1
        overlay_image()
        update_segment_list()

# Function to handle the redo operation (Ctrl-Y)
def redo(event=None):
    global current_index
    if current_index < len(segment_masks) - 1:
        current_index += 1
        overlay_image()
        update_segment_list()

# Function to update the image size display based on DPI input
def update_image_size_display(*args):
    if full_size_image is not None:
        dpi = float(dpi_var.get())
        orig_width, orig_height = full_size_image.size
        width_in_inches = orig_width / dpi
        height_in_inches = orig_height / dpi
        size_display_var.set(f'Image Size: {width_in_inches:.2f} x {height_in_inches:.2f} inches')
        update_segment_list()

# Function to clear the segment list
def clear_segment_list():
    for widget in segment_list_frame.winfo_children():
        widget.destroy()

# Function to update the segment list in the UI
def update_segment_list():
    clear_segment_list()
    dpi = float(dpi_var.get())
    for index, mask in enumerate(segment_masks[:current_index + 1]):
        mask_resized = Image.fromarray(mask).resize(full_size_image.size, Image.Resampling.NEAREST)
        bbox = mask_resized.getbbox()
        if bbox:
            x_size = (bbox[2] - bbox[0]) / dpi
            y_size = (bbox[3] - bbox[1]) / dpi
            color = calculate_color(index, saturation=0.5)
            color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
            
            # Create a frame for each segment entry
            segment_frame = tk.Frame(segment_list_frame, bg=color_hex)
            segment_frame.pack(fill='x', padx=2, pady=2)
            
            # Create a label with the segment number and size
            segment_label = tk.Label(segment_frame, text=f"{index + 1}: {x_size:.2f} x {y_size:.2f} inches", anchor='w', bg=color_hex, fg='black')
            segment_label.pack(fill='x', padx=5, pady=5)

# Create the main window
root = tk.Tk()
root.title("CanvasChopper")
root.geometry("740x567")  # Adjusted size to fit everything comfortably

# Bind the keyboard shortcuts for undo and redo
root.bind("<Control-z>", undo)
root.bind("<Control-y>", redo)

# Set the application icon (replace with the correct path to your icon)
icon_image = tk.PhotoImage(file="canvas-chopper.png")
root.iconphoto(False, icon_image)

# Create the "Load Image" button with a darker gray background
load_button = tk.Button(root, text="Load Image", command=load_image, bg="#555555", fg="white")
load_button.pack(side='left', padx=5, pady=5, anchor='nw')

# Create a flat read-only label with instructions
instruction_label = tk.Label(root, text="Load a jpeg, png, bmp, or gif file.", anchor='w', bg=root.cget("bg"))
instruction_label.pack(side='left', padx=5, pady=8, anchor='nw')

# Create a label for DPI input
dpi_label = tk.Label(root, text="DPI:", anchor='w')
dpi_label.pack(side='left', padx=0, pady=8, anchor='nw')

# Create an entry widget for DPI input
dpi_var = tk.StringVar(value=str(default_dpi))
dpi_entry = tk.Entry(root, textvariable=dpi_var, width=4)
dpi_entry.pack(side='left', padx=0, pady=8, anchor='nw')

# Create a label to display the image size in inches
size_display_var = tk.StringVar(value="Image Size: N/A")
size_display_label = tk.Label(root, textvariable=size_display_var, anchor='w')
size_display_label.pack(side='left', padx=5, pady=8, anchor='nw')

# Create a label widget to display the image, aligned closer to the left
image_frame = tk.Frame(root, width=512, height=512, background='black')
image_frame.place(x=5, y=40, anchor='nw')
image_frame.pack_propagate(0) 
image_label = tk.Label(image_frame, cursor="cross", background='black')
image_label.pack()

# Create a frame for the segment list with a scrollbar
segment_list_canvas = tk.Canvas(root)
segment_list_canvas.place(x=522, y=40, anchor='nw', width=213, height=512)
segment_list_frame = tk.Frame(segment_list_canvas)
segment_list_scrollbar = ttk.Scrollbar(root, orient="vertical", command=segment_list_canvas.yview)
segment_list_scrollbar.place(x=735, y=40, anchor='ne', height=512)
segment_list_canvas.configure(yscrollcommand=segment_list_scrollbar.set)
segment_list_canvas.create_window((0, 0), window=segment_list_frame, anchor='nw')
segment_list_frame.bind("<Configure>", lambda e: segment_list_canvas.configure(scrollregion=segment_list_canvas.bbox("all")))

# Bind the mouse click event to the image label
image_label.bind("<Button-1>", on_image_click)

# Bind changes in DPI input to update the image size display
dpi_var.trace_add("write", update_image_size_display)

# Start the Tkinter event loop
root.mainloop()
