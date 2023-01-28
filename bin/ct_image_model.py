from dicompylercore import dicomparser
import pydicom


class CtImageModel:
    def __init__(self, image_dir):
        self.pydicom = pydicom.read_file(image_dir)
        self.dicomparser = dicomparser.DicomParser(image_dir)
        if hasattr(self.pydicom, 'ImagePositionPatient'):
            self.zlocation = self.pydicom.ImagePositionPatient[2]
        else:
            self.zlocation = None
