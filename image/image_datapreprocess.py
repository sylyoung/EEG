import os
import sys
import pathlib

import pandas as pd
import numpy as np


def create_office31_csv(data_dir, shuffle=False, out_csv_path=None):

    class_names = ['back_pack','bookcase','desk_chair','file_cabinet','laptop_computer','monitor','paper_notebook',
                   'printer','ring_binder','speaker','trash_can','bike','bottle','desk_lamp','headphones','letter_tray',
                   'mouse','pen','projector','ruler','stapler','bike_helmet','calculator','desktop_computer','keyboard',
                   'mobile_phone','mug','phone','punchers','scissors','tape_dispenser']

    image_paths = []
    values = []

    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            f_path = os.path.join(subdir, file)

            if 'DS_Store' in f_path:
                continue
            folder_name = pathlib.PurePath(subdir).name

            for i in range(len(class_names)):
                if class_names[i] == folder_name:
                    class_id = i
                    image_paths.append(f_path)
                    values.append(class_id)

    image_paths = np.array(image_paths).reshape(-1, 1)
    values = np.array(values).reshape(-1, 1)

    concat_arr = np.concatenate([image_paths, values], axis=1)
    if shuffle:
        np.random.shuffle(concat_arr)
    print(concat_arr.shape)
    df = pd.DataFrame(concat_arr)
    df.to_csv(out_csv_path, index=False, header=False)


def create_officehome_csv(data_dir, shuffle=False, out_csv_path=None):

    class_names = ['Alarm_Clock', 'Bottle', 'Chair', 'Desk_Lamp', 'File_Cabinet', 'Glasses', 'Knives', 'Mop', 'Pan',
                   'Printer', 'Scissors', 'Soda', 'ToothBrush', 'Backpack', 'Bucket', 'Clipboards', 'Drill',
                   'Flipflops', 'Hammer', 'Lamp_Shade', 'Mouse', 'Paper_Clip', 'Push_Pin', 'Screwdriver', 'Speaker',
                   'Toys', 'Batteries', 'Calculator', 'Computer', 'Eraser', 'Flowers', 'Helmet', 'Laptop', 'Mug', 'Pen',
                   'Radio', 'Shelf', 'Spoon', 'Trash_Can', 'Bed', 'Calendar', 'Couch', 'Exit_Sign', 'Folder', 'Kettle',
                   'Marker', 'Notebook', 'Pencil', 'Refrigerator', 'Sink', 'Table', 'TV', 'Bike', 'Candle', 'Curtains',
                   'Fan', 'Fork', 'Keyboard', 'Monitor', 'Oven', 'Postit_Notes', 'Ruler', 'Sneakers', 'Telephone',
                   'Webcam']

    image_paths = []
    values = []

    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            f_path = os.path.join(subdir, file)

            if 'DS_Store' in f_path:
                continue
            folder_name = pathlib.PurePath(subdir).name

            for i in range(len(class_names)):
                if class_names[i] == folder_name:
                    class_id = i
                    image_paths.append(f_path)
                    values.append(class_id)

    image_paths = np.array(image_paths).reshape(-1, 1)
    values = np.array(values).reshape(-1, 1)

    concat_arr = np.concatenate([image_paths, values], axis=1)
    if shuffle:
        np.random.shuffle(concat_arr)
    print(concat_arr.shape)
    df = pd.DataFrame(concat_arr)
    df.to_csv(out_csv_path, index=False, header=False)


def create_imageclef_csv(data_dir, shuffle=False, out_csv_path=None):

    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']

    image_paths = []
    values = []

    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            f_path = os.path.join(subdir, file)

            if 'DS_Store' in f_path:
                continue
            folder_name = pathlib.PurePath(subdir).name

            for i in range(len(class_names)):
                if class_names[i] == folder_name:
                    class_id = i
                    image_paths.append(f_path)
                    values.append(class_id)

    image_paths = np.array(image_paths).reshape(-1, 1)
    values = np.array(values).reshape(-1, 1)

    concat_arr = np.concatenate([image_paths, values], axis=1)
    if shuffle:
        np.random.shuffle(concat_arr)
    print(concat_arr.shape)
    df = pd.DataFrame(concat_arr)
    df.to_csv(out_csv_path, index=False, header=False)


def create_visda_csv(path, shuffle=False, out_csv_path=None, root_dir=None):

    image_paths = []
    values = []

    with open(path, 'r') as f:
        for line in f:
            f_path, class_id = line.split(' ')
            f_path = root_dir + f_path
            class_id = int(class_id)

            image_paths.append(f_path)
            values.append(class_id)

    image_paths = np.array(image_paths).reshape(-1, 1)
    values = np.array(values).reshape(-1, 1)

    concat_arr = np.concatenate([image_paths, values], axis=1)
    if shuffle:
        np.random.shuffle(concat_arr)
    print(concat_arr.shape)
    df = pd.DataFrame(concat_arr)
    df.to_csv(out_csv_path, index=False, header=False)


if __name__ == '__main__':
    # Office31
    #path = '/mnt/data2/sylyoung/Image/Office31/OFFICE31/webcam'  # dslr, amazon, webcam
    #out_csv_path = '/mnt/data2/sylyoung/Image/Office31/OFFICE31/W.csv'  # D, A, W
    #create_office31_csv(path, shuffle=True, out_csv_path=out_csv_path)

    # Office-Home
    #path = '/mnt/data2/sylyoung/Image/Office-Home/OfficeHome/RealWorld'  # Art, Clipart, Product, RealWorld
    #out_csv_path = '/mnt/data2/sylyoung/Image/Office-Home/OfficeHome/Re.csv'  # Ar, Cl, Pr, Re
    #create_officehome_csv(path, shuffle=True, out_csv_path=out_csv_path)

    # ImageCLEF
    path = '/mnt/data2/sylyoung/Image/ImageCLEF/image_CLEF/p'  # b, c, i, p
    out_csv_path = '/mnt/data2/sylyoung/Image/ImageCLEF/image_CLEF/p.csv'  # b, c, i, p
    create_imageclef_csv(path, shuffle=True, out_csv_path=out_csv_path)

    # VisDA
    #path = '/mnt/data2/sylyoung/Image/VisDA/validation/image_list.txt'  # train, validation
    #out_csv_path = '/mnt/data2/sylyoung/Image/VisDA/validation.csv'  # train, validation
    #root_dir = '/mnt/data2/sylyoung/Image/VisDA/validation/'  # train, validation
    #create_visda_csv(path, shuffle=True, out_csv_path=out_csv_path, root_dir=root_dir)

