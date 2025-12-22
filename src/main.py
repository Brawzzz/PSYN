import Sample as spl
import setup as stp

from contours_detection import detect_regions

sample = spl.Sample(stp.SAMPLE_INDEX)
sample.set_img_path(n_bf=True)
sample.load_img()

sample.split(nb_split=8)

cntrs = detect_regions(sample)

