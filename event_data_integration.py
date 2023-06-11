import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
import os
import copy


class EventDataProcessor:
    def __init__(self, file_path, first_frame, last_frame):

        print("Importing necessary data...")

        self.file_path = file_path
        self.df = pd.read_excel(file_path, sheet_name='main') # <-- this takes a while
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.frame_start = {}
        self.frame_end = {}
        self.events = self.df

        print("Creating dataframe of events...")

        for i in range(first_frame, last_frame+1):
            frame_start = self.df.loc[self.df['frames'] == i, 'StartTimeStamp'].iloc[0]
            frame_end = self.df.loc[self.df['frames'] == i, 'EndTimeStamp'].iloc[-1]
            self.frame_start[f'{i}'] = frame_start
            self.frame_end[f'{i}'] = frame_end
        self.df = self.concat_columns(self.df)

        print("Initialization done!")

        print("     ‧˚₊•┈┈┈┈୨୧┈┈┈┈•‧₊˚⊹     ")  # cute divider because why not :3

        #selected_rows = self.df[(self.df['x'] == 2) & (self.df['y'] == 2)]
        #print(selected_rows)


    def convert_to_dict(self):
        return self.df.to_dict()
   

    def bool_to_int(self, x):
        if x:
            return 1
        else:
            return -1
   

    def process_frame_events(self, start_time, end_time):
        # events = self.df.loc[((self.df['TimeStamp'] >= start_time) & (self.df['TimeStamp'] <= end_time)) | ((self.df['TimeStamp2'] >= start_time) & (self.df['TimeStamp2'] <= end_time))]
        # events = events.dropna(axis=1)
        # events['Polarity2'] = events['Polarity2'].apply(self.bool_to_int)
        events = self.df.loc[((self.df['TimeStamp'] >= start_time) & (self.df['TimeStamp'] <= end_time))]
        events = events.dropna(axis=1)        
        return events
   

    def concat_columns(self, df):
        new_df = pd.DataFrame()
        new_df['TimeStamp'] = pd.concat([df['TimeStamp'], df['TimeStamp2'], df['TimeStamp3'], df['TimeStamp4'], df['TimeStamp5']], ignore_index=True)
        new_df['y'] = pd.concat([df['y'], df['y2'], df['y3'], df['y4'], df['y5']], ignore_index=True)
        new_df['x'] = pd.concat([df['x'], df['x2'], df['x3'], df['x4'], df['x5']], ignore_index=True)
        new_df['Polarity'] = pd.concat([df['Polarity'], df['Polarity2'], df['Polarity3'], df['Polarity4'], df['Polarity5']], ignore_index=True)
        new_df['Polarity'] = new_df['Polarity'].apply(self.bool_to_int)
        return new_df
    

    def visualize_events(self, df, img_Size):
        """
        Visualizes events from a pandas dataframe as an image using OpenCV.

        Parameters:
            - df (pandas.DataFrame): A pandas dataframe containing event data with columns
                                    'TimeStamp', 'y', 'x', and 'Polarity'.
            - height (int): The height of the output image.
            - width (int): The width of the output image.
            - time_interval (float): The time interval in microseconds over which to accumulate events.

        Returns:
            - numpy.ndarray: The output image as a numpy array of shape (height, width).
        """
        height, width = img_Size
        # time_interval = 1000 # in microseconds

        # Create an empty image
        image = np.zeros((height, width), dtype=np.float32)


        # Loop through each row in the dataframe
        for index, row in df.iterrows():
            
            # Extract the timestamp, y-coordinate, x-coordinate, and polarity from the row
            event_timestamp = row['TimeStamp']
            y = int(row['y'])
            x = int(row['x'])
            polarity = row['Polarity']
            
            # Update the image based on the event polarity and time difference
            if polarity == 1:
                image[y, x] = 1
            elif polarity == -1:
                image[y, x] = -1

        return image
    

    def visualize_events2(self, df, img_Size, time_interval=1000):
        
        height, width = img_Size

        # Create an empty image
        image = np.ones((height, width), dtype=np.float32)

        # Loop through each row in the dataframe
        for index, row in df.iterrows():

            # Extract the y-coordinate, x-coordinate, and polarity from the row
            y = int(row['y'])
            x = int(row['x'])
            polarity = row['Polarity']

            # Update the image based on the event polarity
            image[y, x] = polarity
    
        return image

    def remove_noise(self, df, threshold):
        image_matrix = df.to_numpy()
        processed_matrix = image_matrix.copy()

        rows, cols = image_matrix.shape

        for i in range(rows):
            for j in range(cols):
                if image_matrix[i, j] > threshold:
                    neighbor_sum = 0
                    neighbor_count = 0

                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            ni = i + di
                            nj = j + dj

                            if 0 <= ni < rows and 0 <= nj < cols and image_matrix[ni, nj] <= threshold:
                                neighbor_sum += image_matrix[ni, nj]
                                neighbor_count += 1

                    if neighbor_count > 0:
                        neighbor_mean = neighbor_sum / neighbor_count
                    else:
                        neighbor_mean = 0

                    processed_matrix[i, j] = neighbor_mean

        processed_df = pd.DataFrame(processed_matrix, columns=df.columns, index=df.index)

        return processed_df
    
    import numpy as np

    def shift_image(self, image, n):
        height, width = image.shape

        shifted_image = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                shifted_y = y - n

                if shifted_y >= 0:
                    shifted_image[shifted_y, x] = image[y, x]

        return shifted_image


    
    def integrate(self, first_frame, last_frame):
        first_frame += 1
        last_frame -= 1

        t_shift = -100000

        timescale = 1e6
        c = 0.3

        for frame in range(first_frame, last_frame+1):
             
            print("Integrating blur frame...")
            timestamp_blur = self.frame_start[str(frame)]

            '''
            if frame == first_frame:
                timestamp_after = self.frame_start[str(frame+1)]
                t_after = timestamp_after - timestamp_blur

                event_start = timestamp_blur
                event_end = timestamp_blur + t_after/2

                dT = t_after

            elif frame == last_frame:
                timestamp_before = self.frame_start[str(frame-1)]
                t_before = timestamp_blur - timestamp_before

                event_start = timestamp_blur - t_before/2
                event_end = timestamp_blur

                dT = t_before
            

            #else:
            '''

            timestamp_before, timestamp_after = self.frame_start[str(frame-1)], \
                                                self.frame_start[str(frame+1)]
            
            dt_before = timestamp_blur - timestamp_before
            dt_after = timestamp_after - timestamp_blur

            event_start = (timestamp_blur + t_shift) - dt_before/2 
            event_end = (timestamp_blur + t_shift) + dt_after/2

            dT = dt_after

            summed_image = np.zeros((181, 241), dtype=np.float32)

            slice = 10
            step = (event_end - event_start) / slice
            i = event_start

            while i <= event_end:

                window_frames = data.process_frame_events(i, min(i + step - 1, event_end))
                window_frames['Polarity'] = np.exp(c * window_frames['Polarity'])
                Et = self.visualize_events2(window_frames, (181, 241))
                summed_image += Et

                i += step
            
            #print(summed_image)
            summed_image = self.shift_image(summed_image, 2)

            norm = np.full((181, 241), slice-1, dtype=np.float32)
            summed_image = summed_image - norm
            #replace_zero = np.exp(c)
            #summed_image = np.where(summed_image == 0, replace_zero, summed_image)
            result = ((1/dT * timescale) * summed_image)

            log_image = np.log(result)
            log_image = resize_image(log_image, (240, 178))

            print("Integration done!")

            print("     ‧˚₊•┈┈┈┈୨୧┈┈┈┈•‧₊˚⊹     ")

            print("Subtracting blurred frame from the integrated image...")

            sheet = "frame{}".format(frame)
            blurred_image = pd.read_excel('normalized_sample2.xlsx', sheet_name=sheet)
            log_blur = np.log(blurred_image)


            deblurred_image = np.exp(log_blur - log_image)
            
            #deblurred_image = self.remove_noise(deblurred_image, 0.25)

            print("Unblurring done!")

            plt.imshow(deblurred_image, cmap='gray')
            plt.title("Deblurring with {} Slices and c={}".format(slice, c))
            #plt.colorbar()

            plt.savefig('shift/frame{}.png'.format(frame))

            #plt.show()


    def create_event_video(self, dir_path, video_name, slow_factor=2):

        print("Creating video of events...")

        # Set the output video codec
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # Set the video dimensions based on the event image size
        img = cv2.imread(os.path.join(dir_path, os.listdir(dir_path)[0]))
        height, width, channels = img.shape
        video_size = (width, height)

        # Create a video writer object
        out = cv2.VideoWriter(video_name, fourcc, 5, video_size)

        # Iterate through the event images and write each frame to the output video
        for filename in sorted(os.listdir(dir_path)):
            img_path = os.path.join(dir_path, filename)
            img = cv2.imread(img_path)
            # Write the same frame multiple times to slow down the video
            for i in range(slow_factor):
                out.write(img)

        # Release the video writer object and close all windows
        out.release()
        cv2.destroyAllWindows()

        print("Video done!")

        print("     ‧˚₊•┈┈┈┈୨୧┈┈┈┈•‧₊˚⊹     ") 
    


# Saves a list as a .csv file (just for view purposes :p)
def save_list(list, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(list)


def resize_image(image_matrix, desired_size):

    original_height, original_width = image_matrix.shape

    x_scale = original_width / desired_size[0]
    y_scale = original_height / desired_size[1]

    new_matrix = np.zeros((desired_size[1], desired_size[0]))

    for y in range(desired_size[1]):
        for x in range(desired_size[0]):
            orig_x = int(x * x_scale)
            orig_y = int(y * y_scale)

            new_matrix[y, x] = image_matrix[orig_y, orig_x]

    return new_matrix



if __name__ == '__main__':
    first_frame, last_frame = 5, 7

    data = EventDataProcessor('sample2_data.xlsx', first_frame, last_frame)

    # event_images = []
    # delta_t = 1000
    # start_time = data.frame_start['5'] #+ delta_t/2
    # end_time = data.frame_end['5'] #- delta_t/2
    # print("start_time", start_time)
    # print("end_time", end_time)
    # frame_events = data.process_frame_events(start_time, end_time)
    # temp = data.visualize_events(frame_events, (181, 241))
    # plt.imshow(temp, cmap='gray')
    # plt.show()

    data.integrate(first_frame, last_frame)

    '''

    results = []    

    for i in range(first_frame, last_frame+1):
        start_time = data.frame_start[f"{i}"] 
        end_time = data.frame_end[f"{i}"] 
        frame_events = data.process_frame_events(start_time, end_time)
        if frame_events.empty:
            continue
        else:
            temp = data.visualize_events(frame_events, (181, 241))
            results.append(temp)
    
       
    for i in range(1, len(results)):
        plt.imshow(results[i], cmap='gray')
        plt.savefig(f"{i}")
        print(f"done {i}")

    '''

    print("Code terminated.")

    # data.create_event_video('images', 'event_video.avi')

