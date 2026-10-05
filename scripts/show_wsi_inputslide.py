import openslide
import matplotlib.pyplot as plt

slide = openslide.OpenSlide(
    "data/raw/wsi_data/tcga_blca/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs"
)

thumbnail = slide.get_thumbnail((800, 800))

plt.imshow(thumbnail)
plt.axis("off")
plt.title("Input Whole Slide Image")
plt.show()