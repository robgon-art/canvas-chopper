import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageEnhance, ImageDraw
import onnxruntime as ort
import numpy as np
from copy import deepcopy
import threading
import colorsys
import os
import urllib.request
import cv2
import datetime

def create_menus():
    menu_bar = tk.Menu(root)
    root.config(menu=menu_bar)

    # File menu
    file_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Open Image...", accelerator="Ctrl+O", command=load_image)
    file_menu.add_command(label="Save Segments...", accelerator="Ctrl+S", command=save_image_segments)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", accelerator="Ctrl+Q", command=exit_application)

    # Edit menu
    edit_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Edit", menu=edit_menu)
    edit_menu.add_command(label="Undo", accelerator="Ctrl+Z", command=undo)
    edit_menu.add_command(label="Redo", accelerator="Ctrl+R", command=redo)

    # Bind keyboard shortcuts
    root.bind("<Control-o>", lambda event: load_image())
    root.bind("<Control-s>", lambda event: save_image_segments())
    root.bind("<Control-q>", lambda event: exit_application())
    root.bind("<Control-z>", lambda event: undo())
    root.bind("<Control-r>", lambda event: redo())


def exit_application():
    # Function to cleanly exit the application
    root.destroy()  # Closes the Tkinter window and ends the program

# Download the ONNX model files if they don't exist
def download_file(file_url_base, file_name):
    file_url = file_url_base + file_name
    if not os.path.exists(file_name):
        print(f"Downloading {file_name}...")
        urllib.request.urlretrieve(file_url, file_name)
        print(f"Downloaded {file_name} successfully.")
    else:
        print(f"{file_name} already exists. Skipping download.")
file_url_base = "https://huggingface.co/robgonsalves/segment-anything-8bit-onnx/resolve/main/"
download_file(file_url_base, "sam_encoder.onnx")
download_file(file_url_base, "sam_decoder.onnx")

# Calculate the color based on the index
def calculate_color(index, saturation=1.0, value=1.0):
    hue = (index * phi * 360) % 360
    r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)

# Load the image
def load_image():
    global full_size_image, resized_image, embeddings, segment_masks, scaling_factor, current_index, file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
    if file_path:
        root.config(cursor="watch")
        threading.Thread(target=process_image, args=(file_path,)).start()

# Process the image
def process_image(file_path):
    global full_size_image, resized_image, embeddings, segment_masks, scaling_factor, current_index
    instruction_label.config(text="Running the Segment Anything model.")
    root.update_idletasks()

    # Load the image
    full_size_image = Image.open(file_path).convert("RGB")

    # Resize the image for display
    orig_width, orig_height = full_size_image.size
    new_width = 512
    new_height = int((new_width / orig_width) * orig_height)
    scaling_factor = orig_width / new_width
    resized_image = full_size_image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    update_image_size_display()

    # Clear the segment masks and set the current index
    segment_masks.clear()
    current_index = -1

    # Prepare the image and run the encoder
    input_tensor = np.array(full_size_image)
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    input_tensor = (input_tensor - mean) / std
    input_tensor = input_tensor.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
    pad_height = 1024 - input_tensor.shape[2]
    pad_width = 1024 - input_tensor.shape[3]
    input_tensor = np.pad(input_tensor, ((0, 0), (0, 0), (0, pad_height), (0, pad_width)))
    encoder = ort.InferenceSession("sam_encoder.onnx")
    outputs = encoder.run(None, {"images": input_tensor})
    embeddings = outputs[0]

    # Display the image
    photo = ImageTk.PhotoImage(resized_image)
    image_label.config(image=photo)
    image_label.image = photo
    instruction_label.config(text="Click in the image to define segments.")
    root.config(cursor="arrow")
    clear_segment_list()

def post_process_mask(mask, input_point):
    # Ensure mask is in uint8 format for OpenCV, scaling values to 0-255
    mask_cv = (mask * 255).astype(np.uint8)
    h, w = mask_cv.shape

    # Check if the input_point is within the bounds of the image
    x, y = int(input_point[0][0]), int(input_point[0][1])
    if x >= w or y >= h:
        return mask.astype(np.uint8)  # Ensure we return uint8 mask if point is out of bounds

    # Create an empty image for floodFill output and a mask to constrain floodFill area
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Use floodFill to identify the connected component from the seed point
    cv2.floodFill(mask_cv, flood_mask, (x, y), 255, flags=8 | cv2.FLOODFILL_MASK_ONLY)
    connected_component = flood_mask[1:-1, 1:-1]  # Extract the filled area, removing the border added by floodFill

    # Fill any holes within the connected blob
    contours, _ = cv2.findContours(connected_component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(connected_component, [cnt], -1, 255, -1)

    # Ensuring that the result matches the original image size
    processed_mask = np.zeros_like(mask_cv)
    processed_mask[:h, :w] = (connected_component > 0).astype(np.uint8)

    # Ensure the returned mask is in uint8 format, suitable for use as a PIL image mask
    return processed_mask

# Handle image click events
def on_image_click(event):
    global embeddings, resized_image, segment_masks, scaling_factor, current_index, selection_points, selected_segments

    # Check if the image is loaded
    if embeddings is not None:
        input_point = np.array([[int(event.x * scaling_factor), int(event.y * scaling_factor)]])
        input_label = np.array([1])
        existing_segment_index = -1

        # Check if the point is inside an existing segment
        for index, mask in enumerate(segment_masks[:current_index + 1]):
            mask_x = int(input_point[0][0] * (mask.shape[1] / full_size_image.width))
            mask_y = int(input_point[0][1] * (mask.shape[0] / full_size_image.height))
            if mask[mask_y, mask_x]:
                existing_segment_index = index
                break

        # If the point is inside an existing segment, handle selection logic
        if existing_segment_index != -1:
            handle_selection_logic(event, existing_segment_index)
            overlay_image()
            update_segment_list()
            return

        # Run the decoder to generate the new mask
        onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, np.array([-1])])[None, :].astype(np.float32)
        coords = deepcopy(onnx_coord).astype(float)
        coords[..., 0] = coords[..., 0] * (1024 / full_size_image.width)
        coords[..., 1] = coords[..., 1] * (1024 / full_size_image.height)
        onnx_coord = coords.astype("float32")
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)
        decoder = ort.InferenceSession("sam_decoder.onnx")
        masks_output, _, _ = decoder.run(None, {
            "image_embeddings": embeddings,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array([full_size_image.height, full_size_image.width], dtype=np.float32)
        })

        # Get the new mask and set to uint8
        new_mask = masks_output[0][0]
        new_mask = (new_mask > 0).astype('uint8')

        # Modify the new mask based on existing masks
        for existing_mask in segment_masks[:current_index + 1]:
            new_mask = new_mask & (~existing_mask)

        # Post-process the mask
        new_mask = post_process_mask(new_mask, input_point)

        # Multiply the mask by 255 and add to the segment masks
        new_mask = new_mask * 255
        segment_masks = segment_masks[:current_index + 1]
        selection_points = selection_points[:current_index + 1]
        segment_masks.append(new_mask)
        selection_points.append((input_point[0], (event.x, event.y)))
        current_index += 1

        # Handle selection logic, overlay image, and update segment list
        handle_selection_logic(event, current_index)
        overlay_image()
        update_segment_list()

# Handle selection logic
def handle_selection_logic(event, new_index):
    global selected_segments, shift_down, alt_down
    # SHIFT = 1  # Usually, Shift is represented by the first bit
    # ALT = 512  # Alt is typically represented by the ninth bit
    if shift_down:  # Check if Shift is pressed
        if new_index not in selected_segments:
            selected_segments.append(new_index)
    elif alt_down:  # Check if Alt is pressed
        if new_index in selected_segments:
            selected_segments.remove(new_index)
    else:
        selected_segments = [new_index]

# Add overlay to the image
def overlay_image():
    global resized_image, segment_masks, current_index, selection_points, selected_segments

    # Create and draw the overlay image
    combined_overlay = Image.new("RGBA", resized_image.size, (255, 0, 0, 0))
    for index in range(current_index + 1):
        mask_resized = Image.fromarray(segment_masks[index]).resize(resized_image.size, Image.Resampling.NEAREST)
        color = calculate_color(index)
        colored_overlay = Image.new("RGBA", resized_image.size, color + (128,))
        combined_overlay.paste(colored_overlay, mask=mask_resized)
    combined_image = Image.alpha_composite(resized_image.convert("RGBA"), combined_overlay)
    draw = ImageDraw.Draw(combined_image)

    # Draw the selection points in the overlay image
    for index, (point, screen_coord) in enumerate(selection_points[:current_index + 1]):
        color = calculate_color(index, saturation=0.5)
        outline_color = (0, 0, 255) if index in selected_segments else color
        circle_radius = 10
        x, y = screen_coord
        draw.ellipse((x - circle_radius, y - circle_radius, x + circle_radius, y + circle_radius), fill=color, outline=outline_color, width=2)
        text = str(index + 1)
        text_width, text_height = draw.textbbox((0, 0), text)[2:]
        text_x = x - text_width / 2 + 1
        text_y = y - text_height / 2 + 1
        draw.text((text_x, text_y), text, fill="black")
    
    # Update the image
    photo = ImageTk.PhotoImage(combined_image.convert("RGB"))
    image_label.config(image=photo)
    image_label.image = photo

# Undo function, Ctrl+Z
def undo(event=None):
    global current_index
    if current_index >= 0:
        current_index -= 1
        selected_segments = [max(0, current_index)]
        overlay_image()
        update_segment_list()

# Redo function, Ctrl+Y
def redo(event=None):
    global current_index
    if current_index < len(segment_masks) - 1:
        current_index += 1
        selected_segments = [current_index]
        overlay_image()
        update_segment_list()

# Update the image size display
def update_image_size_display(*args):
    if full_size_image is not None:
        dpi = float(dpi_var.get())
        orig_width, orig_height = full_size_image.size
        width_in_inches = orig_width / dpi
        height_in_inches = orig_height / dpi
        size_display_var.set(f'Image Size: {width_in_inches:.2f} x {height_in_inches:.2f} inches')
        update_segment_list()

# Clear the segment list
def clear_segment_list():
    for widget in segment_list_frame.winfo_children():
        widget.destroy()

# Update the segment list
def update_segment_list():
    global selected_segments
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
            segment_frame = tk.Frame(segment_list_frame, bg=color_hex)
            segment_frame.pack(fill='x', padx=2, pady=2)
            segment_label_text = f"{index + 1}: {x_size:.2f} x {y_size:.2f} inches"
            segment_label = tk.Label(segment_frame, text=segment_label_text, anchor='w', bg=color_hex, fg='black', padx=5, pady=5, bd=0)
            segment_label.pack(fill='x')
            if index in selected_segments:
                segment_label.config(borderwidth=2, relief="flat", highlightbackground="blue", highlightcolor="blue", highlightthickness=2)

import os
import datetime
from tkinter import filedialog
from PIL import Image

def save_image_segments():
    global file_path, full_size_image, segment_masks
    if full_size_image is None or not segment_masks:
        print("No image or segments to save.")
        return

    # Get the base name of the loaded image and prepare the default subfolder name
    image_base_name = os.path.splitext(os.path.basename(file_path))[0]
    now = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    subfolder_name = f"{image_base_name}_{now}"

    # Open a dialog to choose the base directory for saving segments
    base_dir_name = filedialog.askdirectory(title="Select Folder", initialdir=os.path.dirname(file_path))

    if not base_dir_name:
        return  # User cancelled the save operation

    # Construct the full path for the new subfolder
    full_dir_name = os.path.join(base_dir_name, subfolder_name)
    if not os.path.exists(full_dir_name):
        os.makedirs(full_dir_name)  # Create the subfolder if it doesn't exist

    # Save each segment as an image
    for index, mask in enumerate(segment_masks):
        segment_image = Image.new("RGBA", full_size_image.size)
        pixels = segment_image.load()
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x]:
                    r, g, b = full_size_image.getpixel((x, y))
                    pixels[x, y] = (r, g, b, int(mask[y, x] * 255))

        segment_file_path = os.path.join(full_dir_name, f"segment_{index + 1:02}.png")
        segment_image.save(segment_file_path)

    print("Segments saved successfully.")

# Track alt key press and release
def alt_on(event):
    global alt_down
    if not alt_down:
        print("ALT ON")
        image_label.config(cursor="X_cursor")
    alt_down = True

def alt_off(event):
    global alt_down
    if alt_down:
        print("ALT OFF")
        image_label.config(cursor="cross")
    alt_down = False

# track shift key press and release
def shift_on(event):
    global shift_down
    if not shift_down:
        print("SHIFT ON")
        image_label.config(cursor="cross_reverse")
    shift_down = True

def shift_off(event):
    global shift_down
    if shift_down:
        print("SHIFT OFF")
        image_label.config(cursor="cross")
    shift_down = False

# Initialize global variables
full_size_image = None
resized_image = None
embeddings = None
file_path = None
segment_masks = []
selection_points = []
selected_segments = []
scaling_factor = 1.0
phi = 0.61803398875
current_index = -1
default_dpi = 60
alt_down = False
shift_down = False

# Main GUI
root = tk.Tk()
root.title("CanvasChopper")
root.geometry("740x577")
root.bind("<Control-z>", undo)
root.bind("<Control-y>", redo)

# Set the icon
icon_image = tk.PhotoImage(file="canvas-chopper.png")
root.iconphoto(False, icon_image)

# Define the load button
load_button = tk.Button(root, text="Open Image", command=load_image, bg="#555555", fg="white")
load_button.pack(side='left', padx=5, pady=5, anchor='nw')

# Define instruction label
instruction_label = tk.Label(root, text="Load a jpeg, png, bmp, or gif file.", anchor='w', bg=root.cget("bg"))
instruction_label.pack(side='left', padx=5, pady=8, anchor='nw')

# Define the DPI label and entry
dpi_label = tk.Label(root, text="DPI:", anchor='w')
dpi_label.pack(side='left', padx=0, pady=8, anchor='nw')
dpi_var = tk.StringVar(value=str(default_dpi))
dpi_entry = tk.Entry(root, textvariable=dpi_var, width=4)
dpi_entry.pack(side='left', padx=0, pady=8, anchor='nw')
dpi_var.trace_add("write", update_image_size_display)

# Define size display label
size_display_var = tk.StringVar(value="Image Size: N/A")
size_display_label = tk.Label(root, textvariable=size_display_var, anchor='w')
size_display_label.pack(side='left', padx=5, pady=8, anchor='nw')

# Define the image frame and label
image_frame = tk.Frame(root, width=512, height=512, background='black')
image_frame.place(x=5, y=40, anchor='nw')
image_frame.pack_propagate(0)
image_label = tk.Label(image_frame, cursor="cross", background='black')
image_label.pack()
image_label.bind("<Button-1>", on_image_click)

# Define the segment list
segment_list_canvas = tk.Canvas(root)
segment_list_canvas.place(x=522, y=40, anchor='nw', width=213, height=512)
segment_list_frame = tk.Frame(segment_list_canvas)
segment_list_scrollbar = ttk.Scrollbar(root, orient="vertical", command=segment_list_canvas.yview)
segment_list_scrollbar.place(x=735, y=40, anchor='ne', height=512)
segment_list_canvas.configure(yscrollcommand=segment_list_scrollbar.set)
segment_list_canvas.create_window((0, 0), window=segment_list_frame, anchor='nw')
segment_list_frame.bind("<Configure>", lambda e: segment_list_canvas.configure(scrollregion=segment_list_canvas.bbox("all")))

# Bind the Alt and Shift keys
root.bind("<Alt_L>", alt_on)
root.bind("<KeyRelease-Alt_L>", alt_off)
root.bind("<Shift_L>", shift_on)
root.bind("<KeyRelease-Shift_L>", shift_off)

# Create the menus
create_menus()

# Run the main loop
root.mainloop()
