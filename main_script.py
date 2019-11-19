import glob
import os
import sys
import time

VERTICAL_SPACE = 3
DATA_DIR = '/home/thiry/work/ar/photos'
CLOCK_PERIOD = .1
DISPLAY_DOTS = 65
FLUSHING_SPACE = 98

def loop_check(directory=DATA_DIR):
    """Count jpeg files in directory to keep track of new upload."""
    files = glob.glob(os.path.join(directory, "*.jpg"))
    files +=  glob.glob(os.path.join(directory, "*.jpeg"))
    files +=  glob.glob(os.path.join(directory, "*.JPG"))
    nb_files = len(files)
    return nb_files


def wait_file(message):
    index = 0
    nb_files = loop_check()
    while loop_check() == nb_files:
        index = progress_bar(message, index)
    end_progress_bar(index)


def wait_keyboard_interrupt(message):
    index = 0
    try:
        while True:
            index = progress_bar(message, index)
    except KeyboardInterrupt:
        end_progress_bar(index + 2)


def end_progress_bar(index):
    print("." * (DISPLAY_DOTS - index) + "done")


def progress_bar(message, index):
    """
    Progress bar.
    """
    time.sleep(CLOCK_PERIOD)
    if index == DISPLAY_DOTS:
        index -= len(message)
        sys.stdout.write("\b" * index + " " * index + "\b" * index)
        sys.stdout.flush()
        return len(message)
    elif index == 0:
        print(FLUSHING_SPACE * " " + message, end="")
        return len(message)
    else:
        sys.stdout.write(".")
        sys.stdout.flush()
        return index + 1

def reset():
    os.system("clear")
    print("\n" * VERTICAL_SPACE)


if __name__ == "__main__":

    while True:

        reset()

        message = "Press ^C after artists turn"
        wait_keyboard_interrupt(message)

        # open_image(CALIBRATION_IMG)

        message = "Please take a picture"
        wait_file(message)

        message = "separating colors"

        message = "analysing strokes"

        message = "selecting and completing one stroke"

        message = "rendering stroke image to project"

