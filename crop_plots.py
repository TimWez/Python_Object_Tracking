from PIL import Image

def crop_image(image_path, output_path):
    # Open the image
    with Image.open(image_path) as img:
        width, height = img.size

        # Define the crop box (left, upper, right, lower)
        left = 500  # Cropping 300px from the left
        upper = 220  # Cropping 220px from the top
        right = width - 350  # Cropping 250px from the right
        lower = height - 100  # Cropping 100px from the bottom

        # Perform the crop
        cropped_img = img.crop((left, upper, right, lower))

        # Save the cropped image to the output path
        cropped_img.save(output_path)
        print(f"Cropped image saved as {output_path}")

# Example usage


fps_list = [120]                            # Defining the fps list
vid_list = [4,5,6,7,8,9,10,11]          # Defining the video list

for i in range(len(fps_list)):
    for j in range(len(vid_list)):
        crop_image(f"Graphs_verslag_III\\{120}_VID_{vid_list[j]}.png", f"Graphs_verslag_III\\{120}_VID_{vid_list[j]}.png")

#crop_image('afmetingen_2.png', 'afmetingen_3.png')
