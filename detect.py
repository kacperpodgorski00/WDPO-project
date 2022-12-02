import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm

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
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Resizing the image depending on its size
    if ((img.shape[0] < img.shape[1]) or (img.shape[0] == img.shape[1])):
        heightScale = 768 / img.shape[0]
        widthScale = 1020 / img.shape[1]
        resizedImg = cv2.resize(img, [int(img.shape[1] * widthScale), int(img.shape[0] * heightScale)], cv2.INTER_AREA)
    elif ((img.shape[0] > img.shape[1]) or (img.shape[0] == img.shape[1])):
        heightScale = 768 / img.shape[0]
        widthScale = 1020 / img.shape[1]
        resizedImg = cv2.resize(img, [int(img.shape[0] * heightScale), int(img.shape[1] * widthScale)], cv2.INTER_AREA)

    cv2.imshow('window', resizedImg)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    #TODO: Implement detection method.
    
    red = 0
    yellow = 0
    green = 0
    purple = 0

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
