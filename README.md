# Hands-on Perception Project: Deblurring Frames using Event Cameras


## Members of the group:

This project has been carried out by:

* Reem Almheiri
* Hassan Alhosani
* Ahmed Alghfeli


## How to use it:

This project provides a deblurring algorithm that can be used to process and enhance blurred frames. The algorithm requires certain steps to be followed in order to achieve optimal results. This readme file provides instructions on how to use the deblurring algorithm effectively.


## Preparing the Frames:

Before applying the deblurring algorithm, it is recommended to normalize all the frames. To accomplish this, follow these steps:

1. Upload an Excel sheet containing the frames to the `display_image.py` Python file.
2. Use the `normalize_frames` function, which takes as input the Excel sheet with each frame in a separate sheet and the number of frames.
3. Ensure that each sheet is numbered as "frame1, frame2, frame3, ..." and so on. This numbering is necessary for the code to work correctly.


## Applying the Deblurring Algorithm:

To utilize the deblurring algorithm, follow these steps:

1. Open the `event_data_integration.py` Python code.
2. Specify the first frame and the last frame in the `first_frame` and `last_frame` variables, respectively.
3. Create an instance of the `EventDataProcessor` class, providing the name of the normalized Excel sheet, as well as the first and last frames.
4. Call the `integrate` function within the `EventDataProcessor` class, specifying the first and last frames. The progress of the integration will be displayed in the terminal.

Within the `integrate` function, you have the option to customize various parameters according to your requirements:

- `c`: These values can be adjusted to control the intensity of the deblurring effect.
- `t_shift`: This parameter determines the time shift between consecutive frames during the deblurring process.
- `slice`: You can specify the number of slices to be used for the deblurring algorithm.

After the deblurring process is complete, you can choose to view the frames if necessary. Please note that the frames are saved using the `savefig` function, so ensure that you edit the directory before running the code.


## Conclusion

By following the instructions provided in this readme file, you can effectively use the deblurring algorithm to enhance your blurred frames. Experiment with different parameters to achieve the desired results. For further details on the code implementation, please refer to the source files and comments within the code.
