# Image-Processing-and-Edge-Detection-Toolkit
![image](https://github.com/user-attachments/assets/9fd95cc8-7e1f-4bc5-bca1-96410b17c42f)

 Image Processing and Edge Detection Toolkit
This repository contains a collection of image processing techniques and edge detection algorithms implemented using Python. The toolkit provides a graphical user interface (GUI) built with Tkinter, allowing users to load images and apply various image processing operations, including custom implementations of popular edge detection operators.

![image](https://github.com/user-attachments/assets/7f732133-3f82-4b31-8108-79a55889f329)



✨ Features

🧰 Custom Convolution Function:

A general-purpose convolution function that can be used to apply various filters to an image.
✂️ Edge Detection Operators:
🔵 Sobel Operator: Detects edges using horizontal and vertical gradients.
🟢 Prewitt Operator: Another edge detection technique similar to Sobel but with a different kernel.
🟡 Robinson Operator: Detects edges using a specific direction-based kernel.
🔴 Laplacian Operator: Detects edges using the Laplacian of the image, highlighting areas of rapid intensity change.


![image](https://github.com/user-attachments/assets/7ca581f6-d588-4ae3-a395-c9c617166bc3)
![image](https://github.com/user-attachments/assets/c9401c32-afa0-4e37-a18f-3fe1eef24082)

⚙️ Morphological Operations:
🔻 Erosion: Reduces the size of objects in the image, removing noise.
🔺 Dilation: Expands the size of objects, closing small holes.
🔄 Opening: A combination of erosion followed by dilation, used to remove small objects from the foreground.
🔳 Closing: A combination of dilation followed by erosion, used to close small holes in the foreground.


![image](https://github.com/user-attachments/assets/e91a47ed-ef2f-4179-9779-6c3fcc8ead23)

🧩 Image Segmentation:
Segmentation of images based on a threshold value, identifying and highlighting contours.
🎨 Graphical User Interface (GUI):
📂 Load and display images using Tkinter.
⚙️ Apply different image processing techniques and visualize the results side by side using Matplotlib.

![image](https://github.com/user-attachments/assets/73f9d9b3-b44b-428e-81e0-ba61b926c867)

🚀 How to Use
Run the script to launch the GUI.
Load an image using the provided file dialog.
Select and apply the desired image processing technique from the options available.
Visualize the results directly in the GUI or through Matplotlib plots.

![image](https://github.com/user-attachments/assets/5612604c-c0d1-4c0c-88bb-0e183fbf1b0a)

📦 Requirements
Python 3.x
OpenCV
NumPy
Matplotlib
Pillow (PIL)
