# Importing all the necessary libraries
import cv2                          # Used for video analysis
import numpy as np                  # Used for a wide variety of calculations
import math                         # Used for a wide variety of calculations
import matplotlib.pyplot as plt     # Used for creating graphs
import time                         # Used for time related stuff
import csv                          # Used for creating .csv files
import pandas as pd                 # Used for handeling data and .xlsx files
import openpyxl                     # Used for Exel compatibility
from matplotlib import rc           # Used for LaTeX compatibility

# Enable LaTeX-style font rendering
rc('text', usetex=True)
rc('font', family='serif')


px_per_cm = 12                      # The number of pixels that correspond to onm cm in real life

x_pos_list = []                     # Defining the x position list
y_pos_list = []                     # Defining the y position list
velocity_list = []                  # Defining the velocity list
v_list = []

width = []                          # Defining the width of the stone
height = []                         # Defining the height of the stone

fps_list = [120, 240, 960]                            # Defining the fps list
vid_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]    # Defining the video list

distance_list_120 = [4.86, 3.21, 1.96, None, 0.72, 1.63, 0.56, 0.41, 1.61, 1.23, 0.65, 2.52]        # Defining all the distances traveled for 120 fps
distance_list_240 = [1.25, 2.33, None, 2.64, 3.93, 3.92, 1.08, 2.04, 1.43, 0.34, 1.12, 1.78]        # Defining all the distances traveled for 240 fps
distance_list_960 = [1.55, None, 1.29, 4.74, 4.92, 1.18, 3.26, 1.19, 1.17, 5.00, 2.22, 2.46]        # Defining all the distances traveled for 960 fps

m = 0.0127                          # Define m for the friction coeff calculation in kg
g = 9.81                            # Define s for the friction coeff calculation in ms^-2

coeff_list = []                     # Defining the list of friction coefficients

s_time = time.time()                # Defining the start time of the program. Used to calculate the duration of the program

# fps_list_raw = input('Enter FPS list like a b c: ')
# fps_list = [int(x) for x in fps_list_raw.split()]

# vid_nums = input('Enter number of videos per FPS type: ')
# vid_list = []
# for i in range(int(vid_nums)):
#    vid_list.append(i+1)

# file_path = input('Enter filepath to videos: ')

# out_path = input('Enter path for output files: ')
def filter_peaks(lst):
    sorted_list = []
    for i in range(len(lst)):           # Transcripting the list, so the list doesn't get overwritten during the sorting
        sorted_list.append(lst[i])

    sorted_list.sort(reverse=True)      # Sorting the list in descending order (the highest value first)
    print(sorted_list)

    try:
        for i in range(0, len(sorted_list)):
            if sorted_list[i] - sorted_list[i+1] < 0.5:
                if lst.index(sorted_list[i]) < lst.index(sorted_list[0]):
                    pass
                else:
                    return sorted_list[i]
    except IndexError:
        return sorted_list[0]



def write_to_excel(filepath, rows, header):
    """
    :param filepath: Filepath to write the .xlsx file to.                   Example:  filepath = 'Data\\VID_1_120.xlsx'
    :param rows:     zip of lists containing the data.                      Example:  rows = zip(time_list, y_pos_list, velocity_list)
    :param header:   List of strings containing the headers for the data.   Example:  header = ['Time', 'Position', 'Velocity']
    :return:
    """

    df = pd.DataFrame(rows, columns=header)         # Convert the zipped rows into a DataFrame with the given header
    df.to_excel(filepath, index=False)              # Write the DataFrame to an Excel file

    print(f"Excel file '{filepath}' created successfully.")


def csv_file(filepath, action, rows, header):
    """
    :param filepath: Filepath to write the .csv file to.                    Example:  filepath = 'Data\\VID_1_120.xlsx'
    :param action:   Action. 'w' for writing a new file and 'a' for adding values to an existing file
    :param rows:     zip of lists containing the data.                      Example:  rows = zip(time_list, y_pos_list, velocity_list)
    :param header:   List of strings containing the headers for the data.   Example:  header = ['Time', 'Position', 'Velocity']
    :return:         Saves a .csv file to the specified output path.
    """

    with open(filepath, action, newline='') as file:    # Opens or creates a .csv file
        writer = csv.writer(file)

        if action == 'w':                               # Create headers if a new file is made
            writer.writerow(header)

        writer.writerows(rows)                          # Write the data rows


def trace_green_blob(video_path, fps):
    """
    :param video_path:  Filepath to open the .mp4 from.                     Example:  filepath = 'Vids\\VID_1_120.mp4'
    :param fps:         Number of fps (Frames per second) of the video      Example:  fps = 120
    :return:            Function returns lists containing the position of the traced object in the x and y direction in m
                        The velocity of the traced object in m/s
                        The dimensions of the trace object in pixels
    """

    global x_pos_list, y_pos_list, velocity_list, width, height, px_per_cm              # Globalizes previous defined variables

    cap = cv2.VideoCapture(video_path)                  # Capture video from file or camera (use 0 for webcam)

    # Define the range for the green color in HSV
    lower_green = np.array([40, 120, 120])              # Lower bound of green in HSV
    upper_green = np.array([80, 255, 255])              # Upper bound of green in HSV

    origin_x, origin_y = None, None                     # Initialize variables to store the origin (center of the stone in the first frame)

    time_step = 1.0 / fps                               # Time step between frames in seconds

    while True:
        ret, frame = cap.read()                         # Read a frame from the video

        if not ret:
            print("Failed to capture video frame or end of video.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)                                    # Convert frame from BGR to HSV color space

        mask = cv2.inRange(hsv, lower_green, upper_green)                               # Create a mask for green color

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    # Find contours in the mask

        # If any contour is found, trace the largest one
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)        # Find the largest contour by area

            x, y, w, h = cv2.boundingRect(largest_contour)              # Get the bounding rectangle for the largest contour

            width.append(w)         # Append the width to the width list
            height.append(h)        # Append the width to the width list

            # Calculate the center coordinates of the bounding box (the center of the green stone)
            center_x = x + w // 2
            center_y = y + h // 2

            # If this is the first frame, set the origin to the center of the stone
            if origin_x is None and origin_y is None:
                origin_x, origin_y = center_x, center_y
                print(f"Initial origin set at: (X: {origin_x}, Y: {origin_y})")

            # Calculate the relative distance of the stone's center with respect to the origin
            relative_x = ((center_x - origin_x) / px_per_cm) / 100  # 100 for cm to m
            relative_y = (((center_y - origin_y) * -1) / px_per_cm) / 100

            # Print the relative coordinates for each frame
            # print(f"Relative coordinates: (X: {relative_x:.2f}, Y: {relative_y:.2f}) m")

            x_pos_list.append(relative_x)       # Save the x coordinate to the x_pos_list
            y_pos_list.append(relative_y)       # Save the x coordinate to the x_pos_list

            # Calculate velocity (in m/s) if there are at least two points
            if len(x_pos_list) > 1:
                # Calculate the change in position (displacement)
                dx = x_pos_list[-1] - x_pos_list[-2]
                dy = y_pos_list[-1] - y_pos_list[-2]

                # Calculate the Euclidean distance between the points (displacement cm)
                displacement = math.sqrt(dx ** 2 + dy ** 2)

                # Calculate the velocity in m/s: velocity = displacement / time_step
                velocity = displacement / time_step
                velocity_list.append(velocity)

            else:
                # If this is the first frame, the velocity is 0 (since there's no previous frame)
                velocity_list.append(0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)    # Draw a rectangle around the green blob (optional)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)     # Draw a red circle at the center
            cv2.circle(frame, (origin_x, origin_y), 5, (255, 0, 0), -1)     # Draw the origin point as a blue dot (first center) for reference

            # Display the relative coordinates on the frame
            cv2.putText(frame, f"Relative: ({relative_x:.2f}, {relative_y:.2f}) m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display the velocity on the frame (in m/s)
            if len(velocity_list) > 1:
                cv2.putText(frame, f"Velocity: {velocity_list[-1]:.2f} m/s",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        else:
            # Close the window if no contours are found
            print("No contours found. Closing window.")
            break

        cv2.imshow('Green Blob Tracker', frame)     # Display the result frame with the traced blob

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

    return x_pos_list, y_pos_list, velocity_list


def delete_before_after(lst, a=None, b=None):
    """
    Used for cutting of useless pieces for the graph. (When the stone is not moving yet or anymore)
    :param lst: The list on which all the actions are performed. Example: lst = [12,543,12,54]
    :param a:   Integer that corresponds to a position in the list. All the elements before this position will be deleted
    :param b:   Integer that corresponds to a position in the list. All the elements before this position will be deleted
    :return:    Returns a list of which all the elements before position a and after position b are deleted
    """

    # Handle cases where 'a' or 'b' is None
    if a is None:
        a = 0  # Start from the beginning
    if b is None:
        b = len(lst) - 1  # Go to the end of the list

    # Ensure 'a' and 'b' are within valid ranges
    if a < 0 or b >= len(lst) or a > b:
        return []  # Return empty list if invalid positions

    # Slice the list to include only elements between 'a' and 'b' (inclusive)
    return lst[a:b + 1]


def find_list_start(lst, start_offset, end_offset, diff=0.01):
    """
    Used for cutting of useless pieces for the graph. (When the stone is not moving yet or anymore)
    :param lst:             The list on which all the actions are performed. Example: lst = [12,543,12,54]
    :param start_offset:    Integer of a number of frames before the condition is met
    :param end_offset:      Integer of a number of frames before the condition is met
    :param diff:            Integer between 0 and 1.
    :return:                1. Returns the position of the first element that is a diff amount different from that position + 10
                            2. Returns the position of the first element after the start_pos that is a diff amount different from that position + 10
    """
    start_pos = None  # Initialize start_pos
    end_pos = None  # Initialize end_pos

    for i in range(len(lst) - 10):  # Ensure no IndexError by limiting loop range
        if abs(lst[i] - lst[i + 10]) > diff:
            start_pos = i - start_offset
            break

    for i in range(start_pos+start_offset, len(lst) - 10):  # Ensure no IndexError by limiting loop range
        if abs(lst[i] - lst[i + 10]) < diff:
            end_pos = i + end_offset
            break

    return start_pos, end_pos

def calculate_of_list_between_positions(lst, start, end):
    """
    :param lst:     The list on which all the actions are performed. Example: lst = [12,543,12,54]
    :param start:   First position of which the average of the list is taken
    :param end:     Last position of which the average of the list is taken
    :return:        Integer that is the average of a list between position start and position end
    """
    avg = sum(delete_before_after(lst, start, end))/len(delete_before_after(lst, start, end))
    return avg


for i in range(len(fps_list)):
    for j in range(len(vid_list)):
        # Reset lists for each video. If this is not done, the values will all be appended to the same list for every video
        x_pos_list = []
        y_pos_list = []
        velocity_list = []
        width = []
        height = []

        fps = fps_list[i]                               # Defining the fps
        file = f'{fps}/VID_{vid_list[j]}_{fps}.mp4'#f'{fps}/VID_{vid_list[j]}_{fps}.mp4'     # Defining the filepath
        print(file)

        # Reading Vid
        x_pos_list, y_pos_list, velocity_list = trace_green_blob(file, fps)

        # Generate frames list based on FPS
        time_list = [i / fps for i in range(len(y_pos_list))]  # List of times (in seconds)

        # Initializing lists for plotting
        start_pos, end_pos = find_list_start(y_pos_list, 10, 30, 0.01)      # Returns the two positions of in which between the data will be plotted
        print(start_pos, end_pos)

        t_l = delete_before_after(time_list, start_pos, end_pos)            # Reduce the list by deleting the values in which the stone does not move
        v_l = delete_before_after(velocity_list, start_pos, end_pos)        # Reduce the list by deleting the values in which the stone does not move
        y_l = delete_before_after(y_pos_list, start_pos, end_pos)           # Reduce the list by deleting the values in which the stone does not move

        start_area_frame = v_l.index(filter_peaks(v_l)) # xth_biggest_position(v_l, 2)                     # Define the area in which the avg velocity is calculated. This area is 7 frames and starts at the second-biggest value
        end_area_frame = start_area_frame + 6                               # Last value is the size of the area of interest + 1.

        start_area = t_l[start_area_frame]
        try:
            end_area = t_l[end_area_frame]
        except:
            end_area = t_l[-1]

        # Print some values for debugging
        print(f'Num of frames: {len(t_l)}')
        print(f'Start area: {start_area}')
        print(f'End area: {end_area}')
        print(f'biggest position: {start_area_frame}')

        avg_velocity = calculate_of_list_between_positions(v_l, start_area_frame, end_area_frame)       # Calculates the average velocity
        print(f'Average velocity: {avg_velocity}')

        match fps:                                      # Chose the right list with distances
            case 120: distance = distance_list_120
            case 240: distance = distance_list_240
            case 960: distance = distance_list_960
            case _:
                distance = None
                print("Invalid FPS value.")

        v = avg_velocity                                # Define v for the friction coeff calculation in ms^-1
        s = distance[vid_list[j]-1]                     # Define s for the friction coeff calculation in m

        v_list.append(round(v,3))

        if s is None:
            s = 6

        friction_coeff = (0.5*m*v**2)/(m*g*s)           # Calculating the friction coeff
        print(f'Friction coeff: {friction_coeff}')
        print(f'Distance: {s}')

        coeff_list.append(round(friction_coeff,2))               # Append the friction coeff to a list.
                                                        # The mean value and other things can later be calculated over
                                                        # multiple videos from this list

        # Plotting
        fig, ax = plt.subplots(2, figsize=(8, 6))       # Create the figure and axis objects

        # Plot the velocity data
        ax[0].plot(t_l, v_l, color='red', linewidth=2, label='Lijn')         # Plotting the line of the velocity graph

        ax[0].scatter(t_l, v_l, color='red', marker='o', s=4, label='Data')  # Plotting the dots of the velocity graph

        # Add labels and title
        ax[0].set_xlabel('Tijd (s)')
        ax[0].set_ylabel('Snelheid (m/s)')
        ax[0].set_title(f'Nat PO A5D \'Joule voor gevorderden\', file: {file}')

        ax[0].grid()                    # Add a grid to the plot

        # Add shaded area of interest in which the average velocity is calculated
        ax[0].axvspan(start_area, end_area, color='red', alpha=0.25, lw=0, label=f'Gem. snelheid = {round(avg_velocity, 2)} m/s')

        ax[0].legend(loc='upper left')   # Add a legend to the plot

        # Adding a multi-line text box with LaTeX formula in the top right corner\
        textstr = rf'\noindent $s = {s}\ m\\ v = {round(v,3)}\ m/s\\ g = {g}\ m/s^2\\ f = {round(friction_coeff,3)}$'
        plt.text(0.162, 1.8, textstr, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5',
                           alpha=0.5, linewidth=1))



        #########################################

        # Plot the position data
        ax[1].plot(t_l, y_l, color='red', linewidth=2, label='Lijn')         # Plotting the line of the position graph

        ax[1].scatter(t_l, y_l, color='red', marker='o', s=4, label='Snelheid')  # Plotting the dots of the position graph

        # Add labels and title
        ax[1].set_xlabel('Tijd (s)')
        ax[1].set_ylabel('Afstand (m)')

        ax[1].grid()     # Add a grid to the plot

        # Add shaded area of interest in which the average velocity is calculated
        ax[1].axvspan(start_area, end_area, color='red', alpha=0.25, lw=0, label='Interesse gebied')

        # Add a legend to the plot
        ax[1].legend(loc='upper left')

        #plt.show()                         # Display the graph
        graph_file = f'Graphs_vel_new\\{fps}_VID_{vid_list[j]}.png'
        #plt.savefig(graph_file, dpi=500)    # Save the graph
        plt.close()                         # Close the graph

        # Write the data to a .csv data
        rows = zip(time_list, y_pos_list, velocity_list, delete_before_after(v_l, start_area_frame, end_area_frame))
        header = ['Time', 'Position', 'Velocity', 'Avg_Velocity']
        # csv_file(f'VID_{vid_list[j]}_{fps_list[i]}.csv', 'w', rows, header)
        write_to_excel(f'Data\\VID_{vid_list[j]}_{fps_list[i]}.xlsx', rows, header)

b_time = time.time()

print(f'Duration program: {b_time-s_time} seconds')

print(f'Coeff list: {coeff_list}')
print(f'Average Coeff: {sum(coeff_list)/len(coeff_list)}')

print(v_list)
