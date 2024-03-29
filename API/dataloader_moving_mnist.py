import os
import random
import numpy as np
import torch
import torch.utils.data as data
import cv2

from API.ImagesToVideo import img2video


def load_mnist(root):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'train-images.idx3-ubyte')
    with open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist


def load_fixed_set(root):
    # Load the fixed dataset
    filename = 'mnist_test_seq.npy'
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset


class MovingMNIST(data.Dataset):
    def __init__(self, root, is_train=True, n_frames_input=10, n_frames_output=10, num_objects=[2],
                 transform=None):
        super(MovingMNIST, self).__init__()

        self.dataset = None
        if is_train:
            self.mnist = load_mnist(root)
        else:
            if num_objects[0] != 2:
                self.mnist = load_mnist(root)
            else:
                self.dataset = load_fixed_set(root)
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1

        self.mean = 0
        self.std = 1

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames_total, self.image_size_,
                         self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(
                    data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        if self.is_train or self.num_objects[0] != 2:
            # Sample number of objects
            num_digits = random.choice(self.num_objects)
            # Generate data on the fly
            images = self.generate_moving_mnist(num_digits)
        else:
            images = self.dataset[:, idx, ...]

        r = 1
        w = int(64 / r)
        img = np.ones((64, 64, 1))
        img_mask = np.ones((length, 64, 64, 1))
        img_background = np.ones((length, 64, 64, 1))
        for t in range(length):
            img = images[t]
            name = str(t) + '.png'
            file_name = os.path.join("/kaggle/working/front_train/results/mau/video/file", name)
            cv2.imwrite(file_name, img.astype(np.uint8))
        img2video(image_root="/kaggle/working/front_train/results/mau/video/file/", dst_name="/kaggle/working/front_train/results/mau/video/file/images.mp4")
        backSub = cv2.createBackgroundSubtractorMOG2()
        # backSub = cv.createBackgroundSubtractorKNN()
        capture = cv2.VideoCapture(cv2.samples.findFileOrKeep("/kaggle/working/front_train/results/mau/video/file/images.mp4"))
        count = 0
        while True:
            ret, frame = capture.read()
            if frame is None:
                break
            fgMask = backSub.apply(frame)
            fgMask = np.expand_dims(fgMask, axis=2)
            img_mask[count] = fgMask
            background = backSub.getBackgroundImage()
            background_0 = background[:, :, 0]
            background_0 = np.expand_dims(background_0, axis=2)
            img_background[count] = background_0
            count += 1
            # print("count=", count)

        # 20 * 1 * 64 * 64
        images = images.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r * r, w, w))
        images_mask = img_mask.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r * r, w, w))
        images_background = img_background.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape(
            (length, r * r, w, w))
        # img = np.ones((64, 64, 1))
        # capture = cv.VideoCapture(cv.s)

        input = images[:self.n_frames_input]
        input_mask = images_mask[:self.n_frames_input]
        input_background = images_background[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
            output_mask = images_mask[self.n_frames_input:length]
            output_background = images_background[self.n_frames_input:length]
        else:
            output = []

        input = torch.from_numpy(input / 255.0).contiguous().float()
        input_mask = torch.from_numpy(input_mask / 255.0).contiguous().float()
        input_background = torch.from_numpy(input_background / 255.0).contiguous().float()

        output = torch.from_numpy(output / 255.0).contiguous().float()
        output_mask = torch.from_numpy(output_mask / 255.0).contiguous().float()
        output_background = torch.from_numpy(output_background / 255.0).contiguous().float()

        return input_mask, output_mask

    def __len__(self):
        return self.length


def load_data(
        batch_size, val_batch_size,
        data_root, num_workers):

    train_set = MovingMNIST(root=data_root, is_train=True,
                            n_frames_input=10, n_frames_output=10, num_objects=[2])
    test_set = MovingMNIST(root=data_root, is_train=False,
                           n_frames_input=10, n_frames_output=10, num_objects=[2])

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_validation = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, std
