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

from PIL import Image  # Ensure PIL is imported if not already

def select_all_segments():
    global selected_segments, segment_masks
    selected_segments = list(range(len(segment_masks)))
    overlay_image()
    update_segment_list()

def select_no_segments():
    global selected_segments
    selected_segments = []
    overlay_image()
    update_segment_list()

def delete_selected_segments(selected_segments, undoable=True):
    global segment_masks, selection_points
    if selected_segments:
        if undoable:
            push_state()  # Add current state to the stack for undo functionality
        
        # Delete selected segments in reverse order to avoid indexing issues
        for index in sorted(selected_segments, reverse=True):
            del segment_masks[index]
            del selection_points[index]
        
        selected_segments = []
        
        # Update visuals
        overlay_image()
        update_segment_list()

def join_selected_segments():
    global segment_masks, selection_points, selected_segments

    if len(selected_segments) < 2:
        # print("Select at least two segments to join.")
        return
    
    # print(f"Before joining segments: {selected_segments}")
    # for i, mask in enumerate(segment_masks):
    #     area = np.sum(mask)
    #     print(f"  Segment {i}: {area} pixels")
    #     # Save each selected segment as a PNG file
    #     Image.fromarray(mask.astype(np.uint8)).save(f'segment_{i + 1}.png')

    push_state()

    min_index = min(selected_segments)
    combined_mask = np.zeros_like(segment_masks[min_index], dtype=np.uint8)  # Ensure mask is initialized correctly

    for idx in selected_segments:
        combined_mask = np.maximum(combined_mask, segment_masks[idx])

    combined_mask = post_process_mask(combined_mask, selection_points[min_index][0], selection_points[min_index][1])

    print(f"Combined area before post-process: {np.sum(combined_mask)} pixels")

    # Save the combined mask before post-processing
    # Image.fromarray(combined_mask.astype(np.uint8)).save('joined.png')

    new_segment_masks = [mask for i, mask in enumerate(segment_masks) if i not in selected_segments or i == min_index]
    new_segment_points = [point for i, point in enumerate(selection_points) if i not in selected_segments or i == min_index]

    # Replace the segment_masks with updated list before post-processing
    segment_masks = new_segment_masks
    selection_points = new_segment_points

    # Apply post-process to the combined mask
    input_point = np.array(selection_points[min_index], ndmin=2)  # Ensure it is a 2D array
    x, y = input_point[0][0], input_point[0][1]  # Properly extract x and y
    new_mask = post_process_mask(combined_mask, x, y)  # Use x and y in the function call
    segment_masks[min_index] = new_mask
    selected_segments = [min_index]

    # Update visuals
    overlay_image()
    update_segment_list()

# Split the selected segments based on direction
def split_segments(direction):
    assert direction in ("horizontal", "vertical"), "Direction must be 'horizontal' or 'vertical'"

    global segment_masks, selection_points, selected_segments, scaling_factor
    push_state()  # Save the current state before modifying
    segments_to_delete = [] # Initialize list to store segments to delete
    new_segments_count = 0  # Initialize count of new segments

    # Run the loop in reverse to avoid indexing issues
    for index in sorted(selected_segments, reverse=True):
        mask = segment_masks[index]
        h, w = mask.shape
        new_masks = []
        new_points = []

        # Use the selection point to find the blob for this segment
        x, y = selection_points[index]
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(mask.copy(), flood_mask, (x, y), 255, flags=cv2.FLOODFILL_MASK_ONLY)
        blob_mask = flood_mask[1:-1, 1:-1].astype(np.uint8)

        # Calculate the bounding box for the blob
        box_x, box_y, box_w, box_h = cv2.boundingRect(blob_mask)
        
        # Determine the split point based on the direction
        if direction == "horizontal":
            split_point = box_x + box_w // 2
            left_half = blob_mask.copy()
            right_half = blob_mask.copy()
            left_half[:, split_point:] = 0
            right_half[:, :split_point] = 0
            halves = [(left_half, False), (right_half, True)]
        else:  # vertical
            split_point = box_y + box_h // 2
            top_half = blob_mask.copy()
            bottom_half = blob_mask.copy()
            top_half[split_point:, :] = 0
            bottom_half[:split_point, :] = 0
            halves = [(top_half, False), (bottom_half, True)]

        # Process each half to find distinct blobs and create new segments
        for half, is_split_half in halves:
            if half.any():
                contours, _ = cv2.findContours(half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    M = cv2.moments(contour)
                    if M['m00'] == 0:
                        continue  # Skip this contour because its area is zero

                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    new_points.append((cx, cy))
                    mask = np.zeros_like(blob_mask)
                    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                    new_masks.append(mask)
                    new_segments_count += 1  # Increment the count of new segments

        # Keep track of the segments to delete after splitting
        segments_to_delete.append(index)

        # Append the new masks and points
        segment_masks.extend(new_masks)
        selection_points.extend(new_points)

    # Delete the selected segments after splitting
    delete_selected_segments(segments_to_delete, undoable=False)

    # Set the selected_segments to the indices of the newly created segments
    start_index = len(segment_masks) - new_segments_count
    selected_segments = list(range(start_index, len(segment_masks)))

    # Update the visuals
    overlay_image()
    update_segment_list()

# Download the ONNX model files if they don't exist
def download_file(file_url_base, file_name):
    file_url = file_url_base + file_name
    if not os.path.exists(file_name):
        print(f"Downloading {file_name}...")
        urllib.request.urlretrieve(file_url, file_name)
        print(f"Downloaded {file_name} successfully.")
    # else:
    #     print(f"{file_name} already exists. Skipping download.")
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
    global full_size_image, resized_image, embeddings, segment_masks, scaling_factor, file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
    if file_path:
        root.config(cursor="watch")
        threading.Thread(target=process_image, args=(file_path,)).start()

# Process the image
def process_image(file_path):
    global full_size_image, resized_image, embeddings, segment_masks, scaling_factor
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


# Post-process the mask to select the primary component, fill holes, remove noise, etc.
def post_process_mask(mask, input_x, input_y, apply_median=True):
    mask_cv = mask.astype(np.uint8)
    if apply_median:
        mask_cv = cv2.medianBlur(mask_cv, 15)

    h, w = mask_cv.shape

    # Check if the input_point is within the bounds of the image
    x, y = int(input_x), int(input_y)
    if x >= w or y >= h or x < 0 or y < 0:
        return mask_cv  # Return the mask directly if point is out of bounds
    
    # write out mask_cv as a PNG file
    # Image.fromarray(mask_cv).save('mask_cv.png')

    # Initialize floodFill parameters and mask
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(mask_cv, flood_mask, (x, y), 255, flags=cv2.FLOODFILL_MASK_ONLY)
    connected_component = flood_mask[1:-1, 1:-1]

    # Convert values from 1 to 255 where the connected component is
    connected_component = connected_component * 255

    # write out connected_component as a PNG file
    # Image.fromarray(connected_component).save('connected_component.png')

    # Process to fill holes within the connected component
    contours, _ = cv2.findContours(connected_component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(connected_component, [cnt], -1, 255, -1)

    return connected_component  # Return mask in the same format as input

def morphological_filter(amount):
    global segment_masks, selected_segments
    push_state()  # Save the current state before making any changes

    kernel = np.ones((3, 3), np.uint8)
    operation = cv2.dilate if amount > 0 else cv2.erode
    amount = abs(amount)

    # Apply the morphological operation to the selected segments
    selected_masks = {index: operation(segment_masks[index], kernel, iterations=amount)
        for index in selected_segments}

    # Update the global segment_masks with the new non-overlapping selected masks
    for index, mask in selected_masks.items():
        for i, existing_mask in enumerate(segment_masks):
            if i != index:
                mask = mask & (~existing_mask)
        # Apply post-processing to the mask before updating the global segment_masks
        mask = post_process_mask(mask, selection_points[index][0], selection_points[index][1], apply_median=False)
        segment_masks[index] = mask

    overlay_image()
    update_segment_list()

def generate_segment(input_point):
    global embeddings, resized_image, segment_masks, scaling_factor, selection_points, selected_segments

    # Check if the image is loaded
    if embeddings is not None:
        # Prepare input for the ONNX model
        input_point_array = np.array([input_point])  # Convert tuple to 2D array
        input_label = np.array([1])
        onnx_coord = np.concatenate([input_point_array, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, np.array([-1])])[None, :].astype(np.float32)
        coords = deepcopy(onnx_coord).astype(float)
        coords[..., 0] = coords[..., 0] * (1024 / full_size_image.width)
        coords[..., 1] = coords[..., 1] * (1024 / full_size_image.height)
        onnx_coord = coords.astype("float32")
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        # Run the decoder to generate the new mask
        decoder = ort.InferenceSession("sam_decoder.onnx")
        masks_output, _, _ = decoder.run(None, {
            "image_embeddings": embeddings,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array([full_size_image.height, full_size_image.width], dtype=np.float32)
        })

        new_mask = masks_output[0][0]
        new_mask = (new_mask > 0).astype('uint8') * 255  # Scale binary mask to 0 or 255

        # Ensure new_mask is correctly processed against existing masks before and after post-processing
        new_mask = remove_overlaps(new_mask)
        new_mask = post_process_mask(new_mask, input_point[0], input_point[1])
        new_mask = remove_overlaps(new_mask)

        segment_masks.append(new_mask)
        # selection_points.append((input_point[0], (event.x, event.y)))
        selection_points.append(input_point)

def remove_overlaps(new_mask):
    for existing_mask in segment_masks:
        new_mask = new_mask & (~existing_mask)
    return new_mask

def find_segment_at_point(input_point):
    global segment_masks, full_size_image

    # Scale input point coordinates to match the original image dimensions
    x, y = input_point[0], input_point[1]

    # Iterate through all segment masks
    for index, mask in enumerate(segment_masks):
        mask_height, mask_width = mask.shape
        # Scale coordinates to mask size
        mask_x = int(x * (mask_width / full_size_image.width))
        mask_y = int(y * (mask_height / full_size_image.height))
        
        # Check if the scaled point is within the mask
        if mask_y < mask_height and mask_x < mask_width and mask[mask_y, mask_x]:
            return index  # Return the index of the segment if the point is inside

    return -1  # Return -1 if the point is not inside any segment

# Handle image click events
def on_image_click(event):
    global selected_segments, segment_masks, shift_down, alt_down

    # Check if the image is loaded and process the click
    if embeddings is not None:
        # input_point = np.array([[int(event.x * scaling_factor), int(event.y * scaling_factor)]])
        input_point = (int(event.x * scaling_factor), int(event.y * scaling_factor))
        existing_segment_index = find_segment_at_point(input_point)

        # Process based on whether a segment was clicked or not
        if existing_segment_index != -1:
            # Handle modifier keys when clicking on an existing segment
            if shift_down:
                if existing_segment_index not in selected_segments:
                    selected_segments.append(existing_segment_index)
            elif alt_down:
                if existing_segment_index in selected_segments:
                    selected_segments.remove(existing_segment_index)
            else:
                selected_segments = [existing_segment_index]
        else:
            # No existing segment clicked, so possibly create a new segment
            push_state()  # Save current state before modifying
            generate_segment(input_point)
            new_index = len(segment_masks) - 1
            
            # Handle selection of the new segment depending on Shift
            if shift_down:
                selected_segments.append(new_index)
            else:
                selected_segments = [new_index]

        # Update visuals
        overlay_image()
        update_segment_list()

# Add overlay to the image
def overlay_image():
    global resized_image, segment_masks, selection_points, selected_segments

    # Create and draw the overlay image
    combined_overlay = Image.new("RGBA", resized_image.size, (255, 0, 0, 0))
    for index in range(len(segment_masks)):
        mask_resized = Image.fromarray(segment_masks[index]).resize(resized_image.size, Image.Resampling.NEAREST)
        color = calculate_color(index)
        colored_overlay = Image.new("RGBA", resized_image.size, color + (128,))
        combined_overlay.paste(colored_overlay, mask=mask_resized)
    combined_image = Image.alpha_composite(resized_image.convert("RGBA"), combined_overlay)
    draw = ImageDraw.Draw(combined_image)

    # Draw the selection points in the overlay image
    for index, point in enumerate(selection_points):
        x, y = int(point[0] / scaling_factor), int(point[1] / scaling_factor)
        color = calculate_color(index, saturation=0.5)
        outline_color = (0, 0, 255) if index in selected_segments else color
        circle_radius = 10
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
    global segment_masks, selection_points, selected_segments
    if state_stack:
        redo_stack.append({
            "segment_masks": deepcopy(segment_masks),
            "selection_points": deepcopy(selection_points),
            "selected_segments": deepcopy(selected_segments)
        })
        last_state = pop_state()
        segment_masks = last_state["segment_masks"]
        selection_points = last_state["selection_points"]
        selected_segments = last_state["selected_segments"]
        overlay_image()
        update_segment_list()

# Redo function, Ctrl+Y
def redo(event=None):
    global segment_masks, selection_points, selected_segments
    if redo_stack:
        push_state()
        state_to_restore = redo_stack.pop()
        segment_masks = state_to_restore["segment_masks"]
        selection_points = state_to_restore["selection_points"]
        selected_segments = state_to_restore["selected_segments"]
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

def on_segment_label_click(event, index):
    global selected_segments, shift_down, alt_down
    if shift_down:
        if index not in selected_segments:
            selected_segments.append(index)
        else:
            selected_segments.remove(index)
    elif alt_down:
        if index in selected_segments:
            selected_segments.remove(index)
    else:
        selected_segments = [index]
    
    overlay_image()
    update_segment_list()

# Update the segment list
def update_segment_list():
    global selected_segments
    clear_segment_list()
    dpi = float(dpi_var.get())
    for index, mask in enumerate(segment_masks):
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
            segment_label.bind("<Button-1>", lambda event, idx=index: on_segment_label_click(event, idx))
            if index in selected_segments:
                segment_label.config(borderwidth=2, relief="flat", highlightbackground="blue", highlightcolor="blue", highlightthickness=2)

def save_image_segments():
    global file_path, full_size_image, segment_masks
    if full_size_image is None or not segment_masks:
        # print("No image or segments to save.")
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

    # print("Segments saved successfully.")

# Track alt key press and release
def alt_on(event):
    global alt_down
    if not alt_down:
        # print("ALT ON")
        image_label.config(cursor="X_cursor")
    alt_down = True

def alt_off(event):
    global alt_down
    if alt_down:
        # print("ALT OFF")
        image_label.config(cursor="cross")
    alt_down = False

# track shift key press and release
def shift_on(event):
    global shift_down
    if not shift_down:
        # print("SHIFT ON")
        image_label.config(cursor="cross_reverse")
    shift_down = True

def shift_off(event):
    global shift_down
    if shift_down:
        # print("SHIFT OFF")
        image_label.config(cursor="cross")
    shift_down = False

# Functions to manage the state stack for undo and redo
def push_state():
    global state_stack
    # Create a deep copy of the current state and push it onto the stack
    current_state = {
        "segment_masks": deepcopy(segment_masks),
        "selection_points": deepcopy(selection_points),
        "selected_segments": deepcopy(selected_segments)
    }
    state_stack.append(current_state)

def pop_state():
    # Pop the last state from the stack if possible and apply it
    if state_stack:
        return state_stack.pop()
    return None

def exit_application():
    # Function to cleanly exit the application
    root.destroy()  # Closes the Tkinter window and ends the program

def create_menus():
    menu_bar = tk.Menu(root)
    root.config(menu=menu_bar)

    # File menu setup
    file_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Open Image...", accelerator="Ctrl+O", command=load_image)
    file_menu.add_command(label="Save Segments...", accelerator="Ctrl+S", command=save_image_segments)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", accelerator="Ctrl+Q", command=root.quit)

    # Edit menu setup
    edit_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Edit", menu=edit_menu)
    # Typical edit actions
    edit_menu.add_command(label="Undo", accelerator="Ctrl+Z", command=undo)
    edit_menu.add_command(label="Redo", accelerator="Ctrl+R", command=redo)
    edit_menu.add_separator()
    edit_menu.add_command(label="Select All", accelerator="Ctrl+A", command=select_all_segments)
    edit_menu.add_command(label="Select None", accelerator="Ctrl+N", command=select_no_segments)
    edit_menu.add_separator()
    edit_menu.add_command(label="Join Segments", accelerator="Ctrl+J", command=join_selected_segments)
    edit_menu.add_command(label="Delete Segment", accelerator="Ctrl+X", command=lambda: delete_selected_segments(selected_segments))
    edit_menu.add_separator()
    edit_menu.add_command(label="Split Horizontal", accelerator="Ctrl+H", command=lambda: split_segments("horizontal"))
    edit_menu.add_command(label="Split Vertical", accelerator="Ctrl+V", command=lambda: split_segments("vertical"))
    split_type_menu = tk.Menu(edit_menu, tearoff=0)
    edit_menu.add_cascade(label="Split Type", menu=split_type_menu)
    edit_menu.add_separator()
    edit_menu.add_command(label="Grow Segments", accelerator="+", command=lambda: morphological_filter(1))
    edit_menu.add_command(label="Shrink Segments", accelerator="-", command=lambda: morphological_filter(-1))

    # Initialize the dictionary for checkbutton variables
    global split_type_state, split_type
    split_type_state = {}
    split_type = tk.StringVar(value="Straight")  # Default to "Straight"

    # Define the options for split types
    split_type_options = {
        "Straight": "straight",
        "Sine Wave": "sine_wave",
        "Multi-Curve": "multi_curve",
        "Image Contents": "image_contents"
    }

    # Populate split type variables and add checkbuttons dynamically using the default defined in split_type
    for option, split_item in split_type_options.items():
        is_default = (split_item == split_type_options[split_type.get()])
        split_type_state[split_item] = tk.BooleanVar(value=is_default)
        split_type_menu.add_checkbutton(
            label=option,
            onvalue=1,
            offvalue=0,
            variable=split_type_state[split_item],
            command=lambda split_item=split_item, option=option: (split_type_state[split_item].set(True), update_split_type(split_item, option))
        )

    # Bind the keyboard shortcuts to the file menu options
    root.bind("<Control-o>", lambda event: load_image())
    root.bind("<Control-s>", lambda event: save_image_segments())
    root.bind("<Control-q>", lambda event: exit_application())

    # Bind the keyboard shortcuts to the edit menu options
    root.bind("<Control-z>", lambda event: undo())
    root.bind("<Control-r>", lambda event: redo())
    root.bind("<Control-a>", lambda event: select_all_segments())
    root.bind("<Control-n>", lambda event: select_no_segments())
    root.bind("<Control-j>", lambda event: join_selected_segments())
    root.bind("<Control-h>", lambda event: split_segments("horizontal"))
    root.bind("<Control-v>", lambda event: split_segments("vertical"))
    root.bind("<Control-x>", lambda event: delete_selected_segments())
    root.bind("<Delete>", lambda event: delete_selected_segments())
    root.bind("+", lambda event: morphological_filter(1))
    root.bind("-", lambda event: morphological_filter(-1))

# Function to update the selected split type
def update_split_type(selected_split_type, option_name):
    global split_type, split_type_state
    for split_state in split_type_state:
        if split_state != selected_split_type:
            split_type_state[split_state].set(False)
    split_type.set(option_name)  # Update the string variable to the current selection

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
default_dpi = 60
alt_down = False
shift_down = False
state_stack = []
redo_stack = []

# Main GUI
root = tk.Tk()
root.title("CanvasChopper")
root.geometry("740x577")

# Set the icon
icon_image = tk.PhotoImage(file="canvas_chopper.png")
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
