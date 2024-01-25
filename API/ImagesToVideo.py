# coding:utf-8
import os

import cv2

# 获取当前目录
#os.chdir(os.path.split(os.path.realpath(__file__))[0])
#print(os.getcwd())


def is_number(s):
    """判断是不是数字
    """
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def find_continuous_num(astr, c):
    """寻找连续数字
    """
    num = ''
    try:
        while not is_number(astr[c]) and c < len(astr):
            c += 1
        while is_number(astr[c]) and c < len(astr):
            num += astr[c]
            c += 1
    except:
        pass
    if num != '':
        return int(num)


def comp2filename(file1, file2):
    """比较文件名称
    """
    smaller_length = min(len(file1), len(file2))
    for c in range(0, smaller_length):
        if not is_number(file1[c]) and not is_number(file2[c]):
            # print('both not number')
            if file1[c] < file2[c]:
                return True
            if file1[c] > file2[c]:
                return False
            if file1[c] == file2[c]:
                if c == smaller_length - 1:
                    # print('the last bit')
                    if len(file1) < len(file2):
                        return True
                    else:
                        return False
                else:
                    continue
        if is_number(file1[c]) and not is_number(file2[c]):
            return True
        if not is_number(file1[c]) and is_number(file2[c]):
            return False
        if is_number(file1[c]) and is_number(file2[c]):
            if find_continuous_num(file1, c) < find_continuous_num(file2, c):
                return True
            else:
                return False


def sort_insert_filename(file_list):
    """对文件名称进行排序，保证数字的连续性
    """
    for i in range(1, len(file_list)):
        x = file_list[i]
        j = i
        while j > 0 and comp2filename(x, file_list[j - 1]):
            file_list[j] = file_list[j - 1]
            j -= 1
        file_list[j] = x
    return file_list


def img2video(image_root, dst_name, fps=24):
    """将一组图片序列转化为视频文件

    Args:
        image_root (str): 图片序列的文件夹地址
        dst_name ([type]): 输出的视频文件地址
        fps (int, optional): 输出视频序列的帧数. Defaults to 24.
    """
    img_list = os.listdir(image_root)
    img_list = sort_insert_filename(img_list)
    #print(img_list)
    if len(img_list) > 0:
        # 检测图片的长和宽
        img = cv2.imread(os.path.join(image_root, img_list[0]))
        w, h = img.shape[1], img.shape[0]
    else:
        raise Exception("no image in {}".format(image_root))

    fps = fps
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(filename=dst_name, fourcc=fourcc, fps=fps, frameSize=(w, h))

    for img_path in img_list:
        # 逐张写入视频
        path = os.path.join(image_root, img_path)
        if os.path.exists(path):  # 判断图片是否存在
            img = cv2.imread(filename=path)
        else:
            continue
        #cv2.waitKey(10)
        video_writer.write(img)
        #print(img_path + ' done!')
    video_writer.release()


if __name__ == '__main__':
    image_root = './test_img/'
    dst_name = './test_img.mp4'
    fps = 24
    img2video(image_root, dst_name, fps=fps)
    print('finished')

    # 测试文件列表排序
    # print(sort_insert_filename(['a09', 'a2', 'b2', 'a10','a100', 'a01', 'a010', '_a3', 'a893', 'a90']))
