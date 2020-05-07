import os
import cv2
import face_recognition as fg
from PIL import Image, ImageDraw

os.chdir("C:/Users/abdul/OneDrive/Pictures/New folder")
usama_img = fg.load_image_file("usama.jpg")
unknown_img = fg.load_image_file("pic2.jpg")

usama_encoding = fg.face_encodings(usama_img)[0]

known_encoding = [usama_encoding]

known_img_name = ["usama"]

unknown_img_location = fg.face_locations(unknown_img)
unknown_img_encoding = fg.face_encodings(unknown_img, unknown_img_location)

pil_image = Image.fromarray(unknown_img)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)


for (top, left, bottom, right), test_faces in zip(unknown_img_location, unknown_img_encoding):
    name = "unknown"
    matches = fg.compare_faces(known_encoding, test_faces)
    if True in matches:
        first_match_index = matches.index(True)
        name = known_img_name[first_match_index]
        print("found it")

    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom + text_height + 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom + text_height + 5), name, fill=(255, 255, 255, 255))

del draw

# Display the resulting image
pil_image.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
