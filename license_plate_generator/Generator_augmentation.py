import os, random
import cv2
import numpy as np
import argparse

def image_augmentation(img, ang_range=6, shear_range=3, trans_range=3):
    # Rotation
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols, ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 0.9)

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_M = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img, Rot_M, (cols, rows))
    img = cv2.warpAffine(img, Trans_M, (cols, rows))
    img = cv2.warpAffine(img, shear_M, (cols, rows))

    # Brightness
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = .4 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # Blur
    blur_value = random.randint(0,5) * 2 + 1
    img = cv2.blur(img,(blur_value, blur_value))

    return img


class ImageGenerator:
    def __init__(self, save_path):
        self.save_path = save_path
        # Plate
        self.plate = cv2.imread("plate.jpg")

        # loading Number 0-9
        file_path = "./num/"
        file_list = os.listdir(file_path)
        self.Number = list()
        self.number_list = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Number.append(img)
            self.number_list.append(file[0:-4])
            
        # loading Number 1-9 number that start
        file_path = "./numS/"
        file_list = os.listdir(file_path)
        self.NumberS = list()
        self.numberS_list = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.NumberS.append(img)
            self.numberS_list.append(file[0:-4])

        # loading Front Number
        file_path = "./num_front/"
        file_list = os.listdir(file_path)
        self.NumberF = list()
        self.numberF_list = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.NumberF.append(img)
            self.numberF_list.append(file[0:-4])
            
        # loading Char
        file_path = "./char1/"
        file_list = os.listdir(file_path)
        self.char_list = list()
        self.Char1 = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Char1.append(img)
            self.char_list.append(file[0:-4])

        # loading Province
        file_path = "./province/"
        file_list = os.listdir(file_path)
        self.province_list = list() # image name list
        self.Province = list() # image list
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Province.append(img)
            self.province_list.append(file[0:-4])


    def Type_4(self, num, save=False):
        top_w, top_h = 35, 80
        top_char_w, top_char_h = 40, 80
        bot_w, bot_h = 200, 40
        numberF = [cv2.resize(number, (top_w, top_h)) for number in self.NumberF]
        char = [cv2.resize(char1, (top_char_w, top_char_h)) for char1 in self.Char1]
        numberS = [cv2.resize(number, (top_w, top_h)) for number in self.NumberS]
        number2 = [cv2.resize(number, (top_w, top_h)) for number in self.Number]
        province1 = [cv2.resize(province, (bot_w, bot_h)) for province in self.Province]
        char_len = len(char)-1
        province1_len = len(province1)-1
        
        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (330, 150))
            random_width = random.randint(185, 275)
            random_height = random.randint(380, 555)
            random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            background = np.zeros((random_width, random_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (random_height, random_width), (random_R, random_G, random_B), -1)

            label = str()

            # row -> y , col -> x
            row, col = 10, 25

            # number Front
            rand_int = random.randint(0, 9)
            label += self.numberF_list[rand_int]
            Plate[row:row + top_h, col:col + top_w, :] = numberF[rand_int]
            col += top_w

            # character 1
            rand_int = random.randint(0, char_len)
            label += self.char_list[rand_int]   #numbers of char in char1 folder
            Plate[row:row + top_char_h, col:col + top_char_w, :] = char[rand_int]
            col += top_char_w
            
            # character 2
            rand_int = random.randint(0, char_len)
            label += self.char_list[rand_int]
            Plate[row:row + top_char_h, col:col + top_char_w, :] = char[rand_int]
            col += top_w+20

            # number Start
            rand_int = random.randint(0, 8)
            label += self.numberS_list[rand_int]
            Plate[row:row + top_h, col:col + top_w, :] = numberS[rand_int]
            col += top_w

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + top_h, col:col + top_w, :] = number2[rand_int]
            col += top_w

            # number 3
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + top_h, col:col + top_w, :] = number2[rand_int]
            col += top_w

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + top_h, col:col + top_w, :] = number2[rand_int]

            row, col = 97, 65 # new line for province
            
            # province
            rand_int = random.randint(0, province1_len)
            label += self.province_list[rand_int] #77
            Plate[row:row + bot_h, col:col + bot_w, :] = province1[rand_int]

            random_w = random.randint(0, random_width - 150)
            random_h = random.randint(0, random_height - 330)
            background[random_w:150 + random_w, random_h:330 + random_h, :] = Plate
            background = image_augmentation(background)

            if save:
                cv2.imwrite(self.save_path + label + ".jpg", background)
            else:
                cv2.imshow(label, background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img_dir", help="save image directory",
                    type=str, default="../CRNN/DB/")
parser.add_argument("-n", "--num", help="number of image",
                    type=int)
parser.add_argument("-s", "--save", help="save or imshow",
                    type=bool, default=True)
args = parser.parse_args()


img_dir = args.img_dir
A = ImageGenerator(img_dir)

num_img = args.num
Save = args.save

# A.Type_1(num_img, save=Save)
# print("Type 1 finish")
# A.Type_2(num_img, save=Save)
# print("Type 2 finish")
# A.Type_3(num_img, save=Save)
# print("Type 3 finish")
A.Type_4(num_img, save=Save)
print("Type 4 finish")
# A.Type_5(num_img, save=Save)
# print("Type 5 finish")
