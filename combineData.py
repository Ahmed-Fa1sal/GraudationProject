import shutil
import os

folder1 = ''
folder2 = ''
output_folder = ''

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Combine images from the first folder
for filename in os.listdir(folder1):
    image1_path = os.path.join(folder1, filename)
    output_path = os.path.join(output_folder, filename)

    shutil.copy(image1_path, output_path)
    print(f'Copied from folder1: {image1_path}')

# Combine images from the second folder, renaming them to avoid conflicts
for filename in os.listdir(folder2):
    image2_path = os.path.join(folder2, filename)
    output_path = os.path.join(output_folder, filename)

    # Rename the image from the second folder to avoid conflicts
    if os.path.exists(output_path):
        base, extension = os.path.splitext(filename)
        new_filename = base + '_folder2' + extension
        output_path = os.path.join(output_folder, new_filename)

    shutil.copy(image2_path, output_path)
    print(f'Copied from folder2: {image2_path} (renamed)')

print("All images combined successfully.")
