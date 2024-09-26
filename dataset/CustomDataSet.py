from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, df, img_dir_path, extension =".png", transform=None):
        self.df = df
        self.img_dir_path = img_dir_path
        self.extension = extension
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        frame = self.df.iloc[index]
        img_name = frame['id_code']
        label = frame['diagnosis']
        img_path = f"{self.img_dir_path}/{img_name}{self.extension}"
        image = Image.open(img_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label 