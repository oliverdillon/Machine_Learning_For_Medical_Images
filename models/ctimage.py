from dicompylercore import dicomparser
import pydicom
class CTImage:
    def __init__(self,image_dir):
        self.pydicom= pydicom.read_file(image_dir)
        self.dicomparser= dicomparser.DicomParser(image_dir)
        self.zlocation = self.pydicom.ImagePositionPatient[2]