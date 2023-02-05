import json
from pathlib import Path
from typing import Dict

import click
import cv2
import numpy as np
from tqdm import tqdm

# Threshold settings (LT-lower threshold, UT-upper threshold, EDCO-erosion/dilation/closing/opening)
LTYellow = [np.array([20, 153, 100])]
UTYellow = [np.array([26, 255, 255])]
EDCOYellow = np.array([5, 4, 0, 0])

LTGreen = [np.array([36, 195, 137])]
UTGreen = [np.array([51, 255, 245])]
EDCOGreen = np.array([1, 9, 0, 0])

LTRed = [np.array([174, 174, 108])]
UTRed = [np.array([179, 226, 215])]
EDCORed = np.array([3, 10, 0, 0])

LTPurple = [np.array([162, 0, 0])]
UTPurple = [np.array([176, 235, 121])]
EDCOPurple = np.array([0, 0, 13, 0])

def count_candies(hsv_img, lt: list[np.ndarray], ut: list[np.ndarray], edco: list[np.ndarray]):
    numberOfCandies = 0

    while True:
        if hsv_img is None:
            break

        mask = cv2.inRange(hsv_img, lt[0], ut[0])

        erosion = cv2.erode(mask, np.ones((edco[0], edco[0]), np.uint8), iterations=1)
        dilation = cv2.dilate(erosion, np.ones((edco[1], edco[1]), np.uint8), iterations=1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, np.ones((edco[2], edco[2]), np.uint8))
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, np.ones((edco[3], edco[3]), np.uint8))

        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        numberOfCandies = len(contours), opening
        break

    return numberOfCandies


def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """

    # Reading and resizing the image depending on its size
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if ((img.shape[0] < img.shape[1]) or (img.shape[0] == img.shape[1])):
        heightScale = 768 / img.shape[0]
        widthScale = 1020 / img.shape[1]
        resizedImg = cv2.resize(img, [int(img.shape[1] * widthScale), int(img.shape[0] * heightScale)], cv2.INTER_AREA)
    elif ((img.shape[0] > img.shape[1]) or (img.shape[0] == img.shape[1])):
        heightScale = 768 / img.shape[0]
        widthScale = 1020 / img.shape[1]
        resizedImg = cv2.resize(img, [int(img.shape[0] * heightScale), int(img.shape[1] * widthScale)], cv2.INTER_AREA)

    # Removing shadows from objects in the image
    shadowlessImg = cv2.fastNlMeansDenoisingColored(resizedImg, None, 7, 10, 10, 15)
    hsvImg = cv2.cvtColor(shadowlessImg, cv2.COLOR_BGR2HSV)


    #TODO: Implement detection method.
    countedYellow, countedYellowImg = count_candies(hsvImg, LTYellow, UTYellow, EDCOYellow)
    countedGreen, countedGreenImg = count_candies(hsvImg, LTGreen, UTGreen, EDCOGreen)
    countedRed, countedRedImg = count_candies(hsvImg, LTRed, UTRed, EDCORed)
    countedPurple, countedPurpleImg = count_candies(hsvImg, LTPurple, UTPurple, EDCOPurple)


    # print("")
    # print("yellow = " + str(countedYellow))
    # print("green = " + str(countedGreen))
    # print("red = " + str(countedRed))
    # print("purple = " + str(countedPurple))

    #cv2.imshow('orginalImg', resizedImg)
    #cv2.imshow('countedYellow', countedYellowImg)
    #cv2.imshow('countedGreen', countedGreenImg)
    #cv2.imshow('countedRed', countedRedImg)
    #cv2.imshow('countedPurple', countedPurpleImg)


    #cv2.waitKey(0)

    #cv2.destroyAllWindows()


    red = countedRed
    yellow = countedYellow
    green = countedGreen
    purple = countedPurple

    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
