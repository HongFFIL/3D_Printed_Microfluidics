import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
# Set the font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

def distribution_plot():
    # This function will generate Figure 4b
    # Paths to the label folder
    labels_folder = r'C:\Users\delan\Desktop\SPIEManuscript\Figures\figure4gen\labels' # Change to your path

    # Image height (in pixels)
    image_height = 1080

    # Image resolution (µm per pixel)
    resolution_um_per_pixel = 0.345

    # Initialize a list to store y-coordinates in µm
    y_coords_um = []

    # Iterate over all label files in the labels folder
    for label_file in os.listdir(labels_folder):
        if label_file.endswith('.txt'):  # Assuming labels are in .txt files
            label_path = os.path.join(labels_folder, label_file)
            with open(label_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # Ensure the line has the expected format
                        # Extract the normalized y-center coordinate
                        normalized_center_y = float(parts[2])

                        # Convert to absolute pixel values
                        center_y_pixels = normalized_center_y * image_height
                        center_y_um = (image_height - center_y_pixels) * resolution_um_per_pixel

                        # Append to the lists
                        y_coords_um.append(center_y_um)

   # Convert bars to µm
    bar1_um = (image_height - 295) * resolution_um_per_pixel
    bar2_um = (image_height - 903) * resolution_um_per_pixel

    # Calculate the middle line between the bars and shift y-coordinates
    middle_line_um = (bar1_um + bar2_um) / 2
    y_coords_shifted = [y - middle_line_um for y in y_coords_um]

    # Shift the bars as well
    bar1_shifted = bar1_um - middle_line_um
    bar2_shifted = bar2_um - middle_line_um

    # Plot the distribution of y-coordinates
    plt.figure(figsize=(3.75, 3.25))  # Set figure background color
    ax = plt.gca()  # Get current axes
    ax.set_facecolor('#6A6A6A')  # Set axes background color

    plt.hist(y_coords_shifted, bins=50, color='#4233FF', edgecolor='black', alpha=0.7)
    plt.axvline(x=bar2_shifted, color='ghostwhite', linestyle='--', linewidth=1, label=f'y = {bar1_shifted:.0f} µm')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1, label=f'y = 0 µm')
    plt.axvline(x=bar1_shifted, color='ghostwhite', linestyle='--', linewidth=1, label=f'y = {bar2_shifted:.0f} µm')

    # Add titles and labels
    plt.title('Distribution of cell locations', fontsize=12)
    # Italic y in Times New Roman font for the xlabel only
    plt.xlabel(' -coordinate [µm]', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.legend(fontsize=8, loc='best', frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=8)
    # plt.grid(True)
    # Save the figure
    plt.savefig('distribution_plot1.png', dpi=300, bbox_inches='tight')
    # plt.show()

def localize_visualize():
    # This function will generate Figure 4a
    # Paths to the label folder
    labels_folder = r'C:\Users\delan\Desktop\SPIEManuscript\Figures\figure4gen\labels'

    # Image dimensions and resolution
    image_width = 1440  # in pixels
    image_height = 1080  # in pixels
    resolution_um_per_pixel = 0.345  # µm per pixel

    # Initialize lists to store coordinates in µm
    x_coords_um = []
    y_coords_um = []

    # Iterate over all label files in the labels folder
    for label_file in os.listdir(labels_folder):
        if label_file.endswith('.txt'):
            label_path = os.path.join(labels_folder, label_file)
            with open(label_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        normalized_center_x = float(parts[1])
                        normalized_center_y = float(parts[2])

                        # Convert to absolute pixel values and then to µm
                        center_x_pixels = normalized_center_x * image_width
                        center_y_pixels = normalized_center_y * image_height
                        center_x_um = center_x_pixels * resolution_um_per_pixel
                        center_y_um = (image_height - center_y_pixels) * resolution_um_per_pixel

                        # Append to the lists
                        x_coords_um.append(center_x_um)
                        y_coords_um.append(center_y_um)

    # Convert bars to µm
    bar1_um = (image_height - 295) * resolution_um_per_pixel
    bar2_um = (image_height - 903) * resolution_um_per_pixel

    # Calculate the middle line between the bars and shift y-coordinates
    middle_line_um = (bar1_um + bar2_um) / 2
    y_coords_shifted = [y - middle_line_um for y in y_coords_um]

    # Shift the bars as well
    bar1_shifted = bar1_um - middle_line_um
    bar2_shifted = bar2_um - middle_line_um

    sample_image_path = r'C:\Users\delan\Desktop\SPIEManuscript\Figures\figure4gen\background.jpg'

    # Load a sample image as the background
    sample_image = cv2.imread(sample_image_path)

    # Convert BGR (OpenCV) to RGB (matplotlib)
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

    # Plot the image and scatter the points on it
    plt.figure(figsize=(4.5, 3.25))
    plt.imshow(sample_image, extent=[0, image_width * resolution_um_per_pixel, -middle_line_um, image_height * resolution_um_per_pixel - middle_line_um])
    plt.scatter(x_coords_um, y_coords_shifted, color='black', s=5, alpha=0.7, label='cell locations')

    # Add horizontal bars at the shifted y-values
    plt.axhline(y=bar1_shifted, color='ghostwhite', linestyle='--', linewidth=1, label=f'y = {bar1_shifted:.0f} µm')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1, label='y = 0 µm')
    plt.axhline(y=bar2_shifted, color='ghostwhite', linestyle='--', linewidth=1, label=f'y = {bar2_shifted:.0f} µm')

    # Maintain the aspect ratio of the image
    plt.gca().set_aspect('equal', adjustable='box')

    # Add titles and labels
    plt.title('Cell bounding box center locations', fontsize=12)
    plt.xlabel(' -coordinate [µm]', fontsize=10)
    plt.ylabel(' -coordinate [µm]', fontsize=10)
    plt.legend(fontsize=8, frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=8)

    # Save the figure
    plt.savefig('scatter_plot_centered1.png', dpi=300, bbox_inches='tight')

    # Show the plot
    # plt.show()

distribution_plot()
localize_visualize()
