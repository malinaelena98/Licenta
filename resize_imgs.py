from PIL import Image
import os


def resize_image(image, size):
    return image.resize(size, Image.ANTIALIAS)


def resize_images(image_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                try:
                    img = resize_image(img, size)
                    img.save(os.path.join(output_dir, image), img.format)
                except:
                    print("Cannot read image")
        if (i + 1) % 100 == 0:
            print("[{}/{}] Resized the images and saved into '{}'."
                  .format(i + 1, num_images, output_dir))

# image_dir = 'C:\\Users\Home\PycharmProjects\Licenta\dataset2\\train\\malignant'
# output_dir = 'C:\\Users\Home\PycharmProjects\Licenta\\dataset3\\train\\malignant'
# image_dir = 'C:\\Users\Home\PycharmProjects\Licenta\dataset2\\train\\benign'
# output_dir = 'C:\\Users\Home\PycharmProjects\Licenta\\dataset3\\train\\benign'
# image_dir = 'C:\\Users\Home\PycharmProjects\Licenta\dataset2\\valid\\malignant'
# output_dir = 'C:\\Users\Home\PycharmProjects\Licenta\\dataset3\\validation\\malignant'
# image_dir = 'C:\\Users\Home\PycharmProjects\Licenta\dataset2\\valid\\benign'
# output_dir = 'C:\\Users\Home\PycharmProjects\Licenta\\dataset3\\validation\\benign'
# image_dir = 'C:\\Users\Home\PycharmProjects\Licenta\dataset2\\test\\malignant'
# output_dir = 'C:\\Users\Home\PycharmProjects\Licenta\\dataset3\\test\\malignant'
# image_dir = 'C:\\Users\Home\PycharmProjects\Licenta\dataset2\\test\\benign'
# output_dir = 'C:\\Users\Home\PycharmProjects\Licenta\\dataset3\\test\\benign'


# image_size = [224, 224]
# resize_images(image_dir, output_dir, image_size)
