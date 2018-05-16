#!/usr/bin/python3
import os
import sys

from PIL import Image, ImageDraw
import face_recognition as fr


def main():

    known_images = sys.argv[1]
    unknow_images = sys.argv[2]
    output_images = sys.argv[3]

    known_face_encodings = []
    for image_n in os.listdir(known_images):
        image = fr.load_image_file(os.path.join(known_images, image_n))
        encoding = fr.face_encodings(image)[0]
        known_face_encodings.append(encoding)

    for image_n in os.listdir(unknow_images):
        try:
            image = fr.load_image_file(os.path.join(unknow_images, image_n))
            locations = fr.face_locations(image,
                                          number_of_times_to_upsample=0,
                                          model="cnn")
            encodings = fr.face_encodings(image,
                                          known_face_locations=locations)

            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)
            found = False
            for (top, right, bottom, left), encoding in zip(locations,
                                                            encodings):
                if True in fr.compare_faces(known_face_encodings, encoding):
                    draw.rectangle(((left, top), (right, bottom)),
                                   outline=(255, 0, 0))
                    found = True
            del draw
            if found:
                pil_image.save(os.path.join(output_images, image_n))
        except MemoryError:
            sys.stderr.write(
                "Image is too big to process: {0}\n".format(image_n))


if __name__ == "__main__":
    main()
