from __future__ import annotations
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import colorsys

def resize_image(img, max_value):
    """
    Resizes an image while maintaining its aspect ratio, ensuring that neither width nor height exceeds a specified maximum value.

    Parameters:
    - img (numpy.ndarray): The input image to be resized.
    - max_value (int): The maximum value for either width or height after resizing.

    Returns:
    - numpy.ndarray: The resized image.
    """
    if img.shape[0] > max_value:
        shape_ = (int(img.shape[1] * max_value / img.shape[0]), max_value)
        img = cv2.resize(img, shape_)
    return img

class Book_Image():
    def __init__(self, img, name):
        """
        A class for processing images of book spines.

        Parameters:
        - img: numpy.ndarray
            The input image of the book spine.
        - name: str
            A unique identifier for the book image.
        """
        self.name = name
        self.img = img
        self.rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        self.img_width = img.shape[1]
        self.img_height = img.shape[0]
        self.spines = []
        self.spine_separation = []
        self.export_paths = []
        self.spine_pixel_colors = []
    
    def detect_spines(self):
        """
        Detect and separate book spines in the input image using line detection and clustering.
        """
        resized_img = resize_image(self.img, 800)
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        mask = cv2.convertScaleAbs(gray, alpha=0.5, beta=70)
        filtered_lines = []
        fld = cv2.ximgproc.createFastLineDetector()
        lines = fld.detect(gray)
        mask_fld = fld.drawSegments(np.float32(mask), lines)

        cv2.imwrite(f"image_manipulation/{self.name}_gray.png", gray)
        cv2.imwrite(f"image_manipulation/{self.name}_all_lines.png", mask_fld)

        #get vertical lines
        vertical_lines = []
        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                slope = (y2-y1) / (x2-x1)
                if slope > 2 or slope < -2:
                    vertical_lines.append([x1, y1, x2, y2])
        vertical_lines = np.array(vertical_lines)
        vertical_lines = np.resize(vertical_lines, (vertical_lines.shape[0],1,4))

        #get long lines
        long_lines = []
        for i in range(len(vertical_lines)):
            for x1, y1, x2, y2 in vertical_lines[i]:
                length = np.sqrt(np.square(x2-x1) + np.square(y2-y1))
                if length > self.img.shape[0] * 0.07:
                    #long_lines.append([x1, 0, x2, img.shape[0]])
                    long_lines.append([x1, y1, x2, y2])

        long_lines_np = np.array(long_lines)
        long_lines_np = np.resize(long_lines_np, (long_lines_np.shape[0],1,4))

        sorted_points = []
        for i in range(len(long_lines_np)):
            for x1, y1, x2, y2 in long_lines_np[i]:
                if y1 > y2:
                    sorted_points.append([x2, y2, x1, y1])
                else:
                    sorted_points.append([x1, y1, x2, y2])

        sorted_points_np = np.array(sorted_points)
        sorted_points_np = np.resize(sorted_points_np, (sorted_points_np.shape[0],1,4))

        extended_lines = []
        for i in range(len(sorted_points_np)):
            for x1, y1, x2, y2 in sorted_points_np[i]:
                m = (y2-y1) / (x2-x1)
                b = y1 - ( x1 * m )
                y_max = resized_img.shape[0]
                if math.isinf(b):
                    x_min = x1
                    x_max = x2
                else:
                    x_min = - (b / m)
                    x_max = (y_max - b) / m
                extended_lines.append([x_min, 0.0, x_max, y_max])

        extended_lines_np = np.array(extended_lines)
        extended_lines_np = np.resize(extended_lines_np, (extended_lines_np.shape[0],1,4))

        #cluster lines by x-position
        X = [ [x1, x2] for x1, y1, x2, y2 in extended_lines]
        Y = [ [y1, y2] for x1, y1, x2, y2 in extended_lines]

        clusters = []
        for n in range(5,20):
            cluster = KMeans(n_clusters=n, n_init=10).fit(X)
            clusters.append([cluster.inertia_, cluster.cluster_centers_])

        cluster_inertias = [ i[0] for i in clusters ]
        winner_id = min(range(len(cluster_inertias)), key=cluster_inertias.__getitem__)

        winner_centers = clusters[winner_id][1]

        combined_lines = [ [x1, 0.0, x2, self.img.shape[0] ] for x1, x2 in winner_centers ]
        combined_lines = [ [x1, 0.0, x2, self.img.shape[0] ] for x1, x2 in winner_centers ]

        for i,l in enumerate(combined_lines):
            l[1] = Y[i][0]
            l[3] = Y[i][1]

         #sort lines by x-position
        sorted_lines = sorted(combined_lines, key=lambda x: x[0])

        sorted_lines_np = np.array(sorted_lines)
        sorted_lines_np = np.resize(sorted_lines_np, (sorted_lines_np.shape[0],1,4))

        mask_vertical_lines = fld.drawSegments(np.float32(mask), vertical_lines)
        mask_long_lines = fld.drawSegments(np.float32(mask), long_lines_np)
        mask_combined_lines = fld.drawSegments(np.float32(mask), np.float32(sorted_lines_np))
        mask_sorted_lines = fld.drawSegments(np.float32(mask), np.float32(sorted_points_np))
        mask_extended_lines = fld.drawSegments(np.float32(mask), np.float32(extended_lines_np))

        cv2.imwrite(f"image_manipulation/{self.name}_vertical_lines.png", mask_vertical_lines)
        cv2.imwrite(f"image_manipulation/{self.name}_long_lines.png", mask_long_lines)
        cv2.imwrite(f"image_manipulation/{self.name}_combined_lines.png", mask_combined_lines)
        cv2.imwrite(f"image_manipulation/{self.name}_sorted_lines.png", mask_sorted_lines)
        cv2.imwrite(f"image_manipulation/{self.name}_extended_lines.png", mask_extended_lines)

        self.spine_separation = sorted_lines_np

    def extract_spines(self, result_directory, file_prefix):
        """
        Extract individual book spines from the input image and save them as separate images.

        Parameters:
        - result_directory: str
            The directory where the extracted spine images will be saved.
        - file_prefix: str
            A prefix to be added to the names of the saved spine images.
        """
        resized_img = self.resize_image(self.img)
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(resized_img.shape)
        fld = cv2.ximgproc.createFastLineDetector()
        mask_spine_separation = fld.drawSegments(np.float32(mask), np.float32(self.spine_separation))
        cv2.imwrite(f"image_manipulation/{self.name}_rgba_spines.png", mask_spine_separation)

        prev_line = [[0,0,0,self.img.shape[0]]]
        for i, line in enumerate(self.spine_separation):
            mask = np.zeros(resized_img.shape)
            pts = np.array(
                [
                    [prev_line[0][2],prev_line[0][3]],
                    [prev_line[0][0],prev_line[0][1]],
                    [line[0][0],line[0][1]],
                    [line[0][2],line[0][3]]
                ],
                np.int32
            )
            cv2.fillPoly(mask, [pts], color=(255, 255, 255))
            cv2.imwrite(f"image_manipulation/{self.name}_spine_mask.png", mask)
            filt_img = cv2.bitwise_and(np.float32(resized_img), np.float32(mask))
            cv2.imwrite(f"image_manipulation/{self.name}_spine.png", filt_img)

            crop_start = int(min(prev_line[0][0],prev_line[0][2]))
            crop_start = max(0, crop_start)

            crop_end = int(max(line[0][0],line[0][2]))
            crop_end = max(0, crop_end)

            cropped_img = filt_img[ :, crop_start : crop_end ]
            self.spines.append(cropped_img)

            export_path = f"{result_directory}/{file_prefix}{i}.jpg"
            self.export_paths.append(export_path)
            cv2.imwrite(export_path, cropped_img)

            prev_line = line
              

class Book():
    def __init__(self, img, name):
        """
        Initializes a Book object.

        Parameters:
        - img (numpy.ndarray): The image of the book cover.
        - name (str): The name or identifier of the book.
        """
        self.name = name
        self.img = img
        self.resized_img = resize_image(img, 800) #self.resize_image()
        self.img_width = img.shape[0] #book lies on cover, that's why image height is the width of the book
        self.img_height = img.shape[1]
        self.dominant_color = []
        self.export_path = ''

    def detect_dominant_color(self, n_colors=5):
        """
        Detects the dominant color(s) in the resized book cover image using KMeans clustering.

        Parameters:
        - n_colors (int): The number of dominant colors to detect. Default is 5.
        """
        pixels = np.float32(self.resized_img.reshape(-1, 3))
        pixels_black_removed = pixels[~np.all(pixels == [0, 0, 0], axis=1)]
        pixels_black_removed = np.flip(pixels_black_removed, axis=1) #flip order (Images are not stored as RGB but as BGR)
        cluster = KMeans(n_clusters=n_colors, n_init=10).fit(pixels_black_removed)
        max_cluster = max(dict(Counter(cluster.labels_)), key=dict(Counter(cluster.labels_)).get)
        max_center = cluster.cluster_centers_[max_cluster]
        self.dominant_color = max_center

    def extract_spine(self, result_directory, file_prefix):
        """
        Extracts and saves the book spine image.

        Parameters:
        - result_directory (str): The directory where the spine image will be saved.
        - file_prefix (str): The prefix to be used for the saved spine image file.
        """
        export_path = f"{result_directory}/{file_prefix}.jpg"
        self.export_path = export_path
        cv2.imwrite(export_path, self.img)


class Shelving_Unit():
    def __init__(self, shelves_count, shelf_length, books):
        """
        Initializes a Shelving_Unit object.

        Parameters:
        - shelves_count (int): The number of shelves in the shelving unit.
        - shelf_length (int or str): The length of each shelf. If 'auto', it is determined based on the total width of book spines.
        - books (dict): A dictionary of Book objects, where keys are book names.

        Returns:
        - Shelving_Unit: An instance of the Shelving_Unit class.
        """
        self.books = books
        self.shelves_count = shelves_count
        self.shelf_length = self.get_shelf_length(shelf_length)
        self.book_colors = self.get_book_colors()
        self.book_ids = self.get_book_ids()
        self.pca_1d = self.perform_pca()[0]
        self.pca_2d = self.perform_pca()[1]
        
    def get_shelf_length(self, shelf_length):
        """
        Calculates the length of each shelf based on the total width of book spines if 'auto' is specified.

        Parameters:
        - shelf_length (int or str): The length of each shelf. If 'auto', it is determined based on the total width of book spines.

        Returns:
        - int: The calculated shelf length.
        """
        if shelf_length=='auto':
            all_spines_width = sum([i.img.shape[1] for i in self.books.values()])
            shelf_length = all_spines_width / self.shelves_count
        return shelf_length

    def get_book_colors(self):
        """
        Extracts dominant colors from the spines of all books.

        Returns:
        - numpy.ndarray: An array containing the dominant colors of all books in the shelving unit.
        """
        colors = []
        for book in self.books.values():
            colors.append(book.dominant_color)
        
        colors_np = np.array(colors)
        colors_np[colors_np < 0] = 0
        colors_np[colors_np > 255] = 255 

        return colors_np
    
    def get_book_ids(self):
        """
        Retrieves the names of all books in the shelving unit.

        Returns:
        - list: A list containing the names of all books in the shelving unit.
        """
        ids = []
        for book in self.books.values():
            ids.append(book.name)
        return ids

    def perform_pca(self):
        """
        Performs Principal Component Analysis (PCA) on the dominant colors of the book spines.

        Returns:
        - tuple: A tuple containing two arrays representing the 1D and 2D PCA results, respectively.
        """
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(self.book_colors)
        # Normalize values: Lowest x and y value is 0
        x_min = np.min(X_pca_2d, axis=0)[0]
        y_min = np.min(X_pca_2d, axis=0)[1]
        X_pca_2d = X_pca_2d - [x_min, y_min]

        pca_1d = PCA(n_components=1)
        X_pca_1d = pca_1d.fit_transform(self.book_colors)
        x_min = np.min(X_pca_1d, axis=0)[0]
        X_pca_1d = X_pca_1d - [x_min]

        return X_pca_1d, X_pca_2d

    def plot_pca_2d(self, coords=None, colors=None, labels=None, title='2D-PCA of dominant spine colors', filename='pca2d'):
        """
        Plots a 2D PCA visualization of dominant spine colors.

        Parameters:
        - coords (numpy.ndarray): Coordinates for plotting. If None, uses stored PCA coordinates.
        - colors (numpy.ndarray): Colors for plotting. If None, uses stored book colors.
        - labels (list): Labels for data points. If None, uses stored book IDs.
        - title (str): Title of the plot.
        - filename (str): Filename for saving the plot.

        Returns:
        - None
        """
        if coords is None:
            coords = self.pca_2d
            colors = self.book_colors
            labels = self.book_ids
        plt.figure(figsize=(12, 5))
        plt.scatter(coords[:, 0], coords[:, 1], c=colors/255.0, cmap='viridis', edgecolor='k') #marker=(rect_width, rect_height)
        for i, label in enumerate(labels):
            plt.annotate(label, (coords[i, 0], coords[i, 1]))

        plt.title(title)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.savefig(f'result/{filename}.png')
        plt.close()

    def plot_pca_1d(self, coords=None, colors=None, labels=None, title='1D-PCA of dominant spine colors', filename='pca1d'):
        """
        Plots a 1D PCA visualization of dominant spine colors.

        Parameters:
        - coords (numpy.ndarray): Coordinates for plotting. If None, uses stored PCA coordinates.
        - colors (numpy.ndarray): Colors for plotting. If None, uses stored book colors.
        - labels (list): Labels for data points. If None, uses stored book IDs.
        - title (str): Title of the plot.
        - filename (str): Filename for saving the plot.

        Returns:
        - None
        """
        if coords is None:
            coords_ = self.pca_1d
            colors = self.book_colors
            labels = self.book_ids
            x = coords_
            y = np.zeros_like(coords_)
        else:
            x = coords[:, 0]
            y = coords[:, 1]
        plt.figure(figsize=(20, 4))
        plt.scatter(x, y, c=colors/255.0, cmap='viridis', edgecolor='k')
        for i, label in enumerate(labels):
            if coords is None:
                annotation_positions = (coords_[i],0)
            else:
                annotation_positions = (coords[i, 0], coords[i, 1])
            plt.annotate(label, annotation_positions)

        plt.title(title)
        plt.xlabel('Principal Component 1')
        plt.savefig(f'result/{filename}.png')
        plt.close()

    def plot_combined_pca(self):
        """
        Plots a combined visualization of 2D and 1D PCA of dominant spine colors.

        Returns:
        - None
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].scatter(self.pca_2d[:, 0], self.pca_2d[:, 1], c=self.book_colors/255.0, cmap='viridis', edgecolor='k')
        for i, label in enumerate(self.book_ids):
            axs[0].annotate(label, (self.pca_2d[i, 0], self.pca_2d[i, 1]))

        axs[0].set_title('2D-PCA of dominant spine colors')
        axs[0].set_xlabel('Principal Component 1')
        axs[0].set_ylabel('Principal Component 2')

        axs[1].scatter(self.pca_1d, np.zeros_like(self.pca_1d), c=self.book_colors/255.0, cmap='viridis', edgecolor='k')
        for i, label in enumerate(self.book_ids):
            axs[1].annotate(label, (self.pca_1d[i], 0))

        axs[1].set_title('1D-PCA of dominant spine colors')
        axs[1].set_xlabel('Principal Component 1')
        
        plt.tight_layout()
        plt.savefig('result/combined_pca.png')
        plt.close()

    def sort_books_2d(self):
        """
        Sorts books and arranges them on shelves based on 2D PCA positions of dominant spine colors.

        Returns:
        - None
        """
        X_pca = self.pca_2d
        shelves = [] #this 2D-list will contain a list of spines for each shelf, index 0 is lowest shelf
        remaining_space = [] #this 1D-list will contain the remaining space in pixels for each shelf, index 0 is lowest shelf
        #y_pos_shelves = []
        final_positions = []
        final_labels = []
        final_colors = []
        max_y = (np.max(X_pca, axis=0)[1]) / self.shelves_count

        for shelf in range(self.shelves_count):
            shelves.append([])
            remaining_space.append(self.shelf_length)
            y_shelf_delta = max_y #/ count_shelves
            y_pos_shelves = [ y_shelf_delta * i for i in range(self.shelves_count) ]

        # Fill book shelves
        count_books = 0
        total_books = len(self.books) #len(os.listdir('detected_spines/'))

        for shelf in range(self.shelves_count):
            current_x = 0 #position on x-axis where the next book should be placed
            current_y = y_pos_shelves[shelf]
            while remaining_space[shelf] > 0:
                
                shortest_distance = 999999
                for i, spine_position in enumerate(X_pca):
                    this_distance = math.dist([ current_x, current_y * (X_pca.shape[0] / X_pca.shape[1])],spine_position)
                    if this_distance <= shortest_distance:
                        shortest_distance = this_distance
                        nearest_spine_id = self.books[i].name
                        nearest_spine_pca_id = i
                
                img_nearest_spine = self.books[nearest_spine_id].img #cv2.imread('detected_spines/'+str(nearest_spine_id)+'.jpg') 
                spine_width = img_nearest_spine.shape[1] #img_nearest_spine.shape[0]
                shelves[shelf].append(nearest_spine_id)
                remaining_space[shelf] -= spine_width
                current_x += spine_width
                count_books += 1
                if current_x >= self.shelf_length:
                    break
                elif count_books >= total_books:
                    break
                else:
                    final_positions.append([current_x, current_y])
                    final_labels.append(nearest_spine_id)
                    final_colors.append(self.book_colors[nearest_spine_pca_id])
                    X_pca[nearest_spine_pca_id] = np.nan

        final_positions = np.array(final_positions)
        final_colors = np.array(final_colors)

        if len(final_positions)>0:
            self.plot_pca_2d(
                colors = final_colors, 
                coords = final_positions, 
                labels = final_labels, 
                title = f'Rearranged 2D-PCA', 
                filename = f'rearranged_pca2d'
            )

        #create image of shelving unit
        shelf_images = []
        for shelf in shelves:
            shelf_img = np.zeros(self.books[0].img.shape, dtype=np.float32)
            for book in shelf:
                book_id = self.book_ids[book]
                shelf_img = cv2.hconcat([shelf_img, self.books[book_id].img])
            shelf_images.append(shelf_img)
        
        shelving_unit_img = self.stack_images_vertically(shelf_images)
        cv2.imwrite(f"result/shelving_unit_2d.png", shelving_unit_img)

    def stack_images_vertically(self, img_list):
        """
        Stacks a list of images vertically to create a single image.

        Parameters:
        - img_list (list): List of images to be stacked vertically.

        Returns:
        - numpy.ndarray: The stacked image.
        """
        max_width = 0
        total_height = 200  # padding
        for img in img_list:
            if img.shape[1] > max_width:
                max_width = img.shape[1]
            total_height += img.shape[0]

        # create a new array with a size large enough to contain all the images
        final_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)

        current_y = 0  # keep track of where your current image was last placed in the y coordinate
        for image in img_list:
            # add an image to the final array and increment the y coordinate
            image = np.hstack((image, np.zeros((image.shape[0], max_width - image.shape[1], 3))))
            final_image[current_y:current_y + image.shape[0], :, :] = image
            current_y += image.shape[0]
        return final_image

    def sort_books_1d(self):
        """
        Sorts books and arranges them on shelves based on 1D PCA positions of dominant spine colors.

        Returns:
        - None
        """
        X_pca = self.pca_1d
        shelves = [] #this 2D-list will contain a list of spines for each shelf, index 0 is lowest shelf
        remaining_space = [] #this 1D-list will contain the remaining space in pixels for each shelf, index 0 is lowest shelf
        #y_pos_shelves = []
        final_positions = []
        final_labels = []
        final_colors = []
        max_y = 100 #(np.max(X_pca, axis=0)[1]) / self.shelves_count

        for shelf in range(self.shelves_count):
            shelves.append([])
            remaining_space.append(self.shelf_length)
            y_shelf_delta = max_y #/ count_shelves
            y_pos_shelves = [ y_shelf_delta * i for i in range(self.shelves_count) ]

        #sort books by PCA position
        pos_labels_colors = []
        for i, position in enumerate(X_pca):
            pos_labels_colors.append({
                'position' : position[0],
                'label'    : i,
                #'color'    : self.book_colors[i]
            })
        sorted_books = sorted(pos_labels_colors, key=lambda d: d['position']) 

        for book in sorted_books:
            book_img = self.books[book['label']].img
            spine_width = book_img.shape[1]

            emptiest_shelf = max(range(len(remaining_space)), key=remaining_space.__getitem__)

            shelves[emptiest_shelf].append(book['label'])
            remaining_space[emptiest_shelf] -= spine_width

            final_positions.append([self.shelf_length - remaining_space[emptiest_shelf], y_pos_shelves[emptiest_shelf]])
            final_labels.append(book['label'])
            final_colors.append(self.book_colors[book['label']])

        final_positions = np.array(final_positions)
        final_colors = np.array(final_colors)

        if len(final_positions)>0:
            self.plot_pca_1d(
                colors = final_colors, 
                coords = final_positions, 
                labels = final_labels, 
                title = f'Rearranged 1D-PCA', 
                filename = f'rearranged_pca1d'
            )

        #create image of shelving unit
        shelf_images = []
        for shelf in shelves:
            shelf_img = np.zeros(self.books[0].img.shape, dtype=np.float32)
            for book in shelf:
                book_id = self.book_ids[book]
                shelf_img = cv2.hconcat([shelf_img, self.books[book_id].img])
            shelf_images.append(shelf_img)
        
        shelving_unit_img = self.stack_images_vertically(shelf_images)
        cv2.imwrite(f"result/shelving_unit_1d.png", shelving_unit_img)

    def sort_books_hsv(self):
        """
        Sorts books and arranges them on shelves based on HSV values of dominant spine colors.

        Returns:
        - None
        """
        X_pca = self.pca_1d
        shelves = [] #this 2D-list will contain a list of spines for each shelf, index 0 is lowest shelf
        remaining_space = [] #this 1D-list will contain the remaining space in pixels for each shelf, index 0 is lowest shelf
        #y_pos_shelves = []
        final_positions = []
        final_labels = []
        final_colors = []
        max_y = 100 #(np.max(X_pca, axis=0)[1]) / self.shelves_count

        for shelf in range(self.shelves_count):
            shelves.append([])
            remaining_space.append(self.shelf_length)
            y_shelf_delta = max_y #/ count_shelves
            y_pos_shelves = [ y_shelf_delta * i for i in range(self.shelves_count) ]

        #sort books by PCA position
        pos_labels_colors = []
        for i, color in enumerate(self.book_colors):
            pos_labels_colors.append({
                'label'    : i,
                'hsv'    : colorsys.rgb_to_hsv(color[0],color[1],color[2])
            })
        sorted_books = sorted(pos_labels_colors, key=lambda d: d['hsv'])

        for book in sorted_books:
            book_img = self.books[book['label']].img
            spine_width = book_img.shape[1]

            emptiest_shelf = max(range(len(remaining_space)), key=remaining_space.__getitem__)

            shelves[emptiest_shelf].append(book['label'])
            remaining_space[emptiest_shelf] -= spine_width

            final_positions.append([self.shelf_length - remaining_space[emptiest_shelf], y_pos_shelves[emptiest_shelf]])
            final_labels.append(book['label'])
            final_colors.append(self.book_colors[book['label']])

        final_positions = np.array(final_positions)
        final_colors = np.array(final_colors)

        if len(final_positions)>0:
            self.plot_pca_1d(
                colors = final_colors, 
                coords = final_positions, 
                labels = final_labels, 
                title = f'Rearranged 1D-PCA', 
                filename = f'rearranged_pca1d'
            )

        #create image of shelving unit
        shelf_images = []
        for shelf in shelves:
            shelf_img = np.zeros(self.books[0].img.shape, dtype=np.float32)
            for book in shelf:
                book_id = self.book_ids[book]
                shelf_img = cv2.hconcat([shelf_img, self.books[book_id].img])
            shelf_images.append(shelf_img)
        
        shelving_unit_img = self.stack_images_vertically(shelf_images)
        cv2.imwrite(f"result/shelving_unit_hsv.png", shelving_unit_img)


def main():
    # Load and process book images
    book_images = {}
    for root, dirs, files in os.walk('img/'):
        files_pbar = tqdm(files)
        for i, file in enumerate(files_pbar):
            files_pbar.set_description("Analyse book images")
            img = cv2.imread(os.path.join(root, file))
            book_image = Book_Image(img, name=i)
            book_image.detect_spines()
            book_images[i] = book_image
            book_image.extract_spines(result_directory='detected_spines', file_prefix=f'{i}_')

    # Load books and identify dominant colors
    books = {}
    i=0
    files_pbar = tqdm(book_images.values())
    for book_img in tqdm(book_images.values()):
        files_pbar.set_description("Load books")
        for spine in book_img.spines:
            book = Book(spine, name=i)
            book.detect_dominant_color(n_colors=20)
            books[i] = book
            book.extract_spine(result_directory='detected_spines_books', file_prefix=f'{i}')
            i += 1

    # Create Book Shelving Unit and sort books
    shelving_unit = Shelving_Unit(
        shelves_count = 5,
        shelf_length = 'auto',
        books = books
    )

    shelving_unit.plot_pca_1d()
    shelving_unit.plot_pca_2d()
    shelving_unit.plot_combined_pca()
    shelving_unit.sort_books_1d()
    shelving_unit.sort_books_2d()
    shelving_unit.sort_books_hsv()

    print('')

if __name__ == "__main__":
    main()
