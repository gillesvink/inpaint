import time
import inpaint
from PIL import Image

def main():
    image = Image.open("../../test/images/baked/toad.png")
    mask = Image.open("../../test/images/mask/text.png")

    start_time = time.time()
    output = inpaint.telea(image, mask, 5)
    end_time = time.time()

    output.save("./output.png")

    elapsed_time = end_time - start_time
    print(f"Inpainting finished in {elapsed_time:.2f} second(s)")

if __name__ == "__main__":
    main()
