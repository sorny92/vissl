import sys

# setting path
sys.path.append('../')
from dataset.DVCLoader import DVCDataset
from dataset.augmentations import transforms, light_transforms, resize
from vissl.data.data_helper import get_mean_image
from torch.utils.data import Dataset


def get_crop_label_data(pkl_path):
    import pickle
    with open(pkl_path, "rb") as f:
        crop_label_data = pickle.load(f)
    return crop_label_data


class DVCswavDataset(Dataset):
    """
    add documentation on how this dataset works

    Args:
        add docstrings for the parameters
    """

    def __init__(self, cfg, data_source, path, split, dataset_name):
        super(DVCswavDataset, self).__init__()
        assert data_source in [
            "disk_filelist",
            "disk_folder",
            "DVCswavDataset"
        ], "data_source must be either disk_filelist or disk_folder or my_data_source"
        self.cfg = cfg
        self.split = split
        self.dataset_name = dataset_name
        self.data_source = data_source
        self._path = path
        self.dvc_dataset = DVCDataset(cfg["DATA"]["internal_data_path"],
                                      cfg["DATA"]["external_data_path"],
                                      cfg["DATA"]["ambient_data_path"],
                                      cfg["DATA"]["cluster_family_path"],
                                      resize, None, None,
                                      get_crop_label_data(cfg["DATA"]["crop_data_path"]))
        # implement anything that data source init should do
        self._num_samples = self.dvc_dataset.__len__()

    def num_samples(self):
        """
        Size of the dataset
        """
        return self.dvc_dataset.__len__()

    def __len__(self):
        """
        Size of the dataset
        """
        return self.dvc_dataset.__len__()

    def __getitem__(self, idx: int):
        """
        implement how to load the data corresponding to idx element in the dataset
        from your data source
        """
        loaded_data = self.dvc_dataset.__getitem__(idx)
        return loaded_data, True
