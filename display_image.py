import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

def find_max_min(file_path, no_of_frames):

    max_val = -np.Inf
    min_val = np.Inf

    for i in range(1, no_of_frames+1):
        sheet = "frame{}".format(i)
        df = pd.read_excel(file_path, sheet_name=sheet)
        max_val = max(max_val, df.max().max())
        min_val = min(min_val, df.min().min())
    return (max_val, min_val)

def normalize_frames(file_path, no_of_frames):

    max_val, min_val = find_max_min(file_path, no_of_frames)

    with pd.ExcelWriter('normalized_sample2.xlsx') as writer:
        for i in range(1, no_of_frames+1):
            sheet = "frame{}".format(i)
            df = pd.read_excel(file_path, sheet_name=sheet)
            df_norm = (df - min_val) / (max_val - min_val) 
            df_norm.to_excel(writer, sheet_name=sheet, index=False, header=False)
    
    print("Excel sheet created successully!")


# Create normalized frames
normalize_frames('sample2_data.xlsx', 10)


# Find mix/max values
'''
max_val, min_val = find_max_min('sample2_data.xlsx', 10)
print("The max value between all frames is", max_val, "and the min value is", min_val, ".")
'''


# Display latent images
'''
first_frame, last_frame = 1, 10
for frame in range(first_frame, last_frame+1):
    
    df = pd.read_excel('normalized_sample2.xlsx', sheet_name='frame{}'.format(frame))
    img = df.values

    plt.imshow(img, cmap='gray')
    #plt.colorbar()
    plt.title('Normalized Image')
    plt.savefig('before/frame{}.png'.format(frame))
    #plt.show()

'''
