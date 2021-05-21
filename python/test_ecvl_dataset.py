import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

yaml_path = "../../../datasets/BIMCV-COVID19-cIter_1_2/covid19_posi/ecvl_bimcv_covid19+.yaml"
batch_size = 16
H, W = 512, 512

tr_augs = ecvl.SequentialAugmentationContainer([
    ecvl.AugResizeDim((H, W), ecvl.InterpolationType.cubic)
])

val_augs = ecvl.SequentialAugmentationContainer([
    ecvl.AugResizeDim((H, W), ecvl.InterpolationType.cubic)
])

te_augs = ecvl.SequentialAugmentationContainer([
    ecvl.AugResizeDim((H, W), ecvl.InterpolationType.cubic)
])

dataset_augs = ecvl.DatasetAugmentations([tr_augs, val_augs, te_augs])

dataset = ecvl.DLDataset(yaml_path, batch_size, dataset_augs, ecvl.ColorType.GRAY)

num_classes = len(dataset.classes_)
print(f"Classes: {dataset.classes_}")

x = Tensor([batch_size, dataset.n_channels_, H, W])
y = Tensor([batch_size, num_classes])

dataset.SetSplit(ecvl.SplitType.training)

dataset.LoadBatch(x, y)

x.info()
print(f"mean: {x.mean()} - max: {x.max()} - min: {x.min()}")

y.info()
y.print()
