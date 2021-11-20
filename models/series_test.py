import unittest
import os

from models.series import Series


class TestSeries(unittest.TestCase):
    def setUp(self):
        self.read_data_titles = "Series UID,Collection,3rd Party Analysis,Data Description URI,Subject ID,Study UID,Study Description,Study Date,Series Description,Manufacturer,Modality,SOP Class Name,SOP Class UID,Number of Images,File Size,File Location,Download Timestamp"
        
        self.test_read_data = [
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.150602874655732880945154527032,HNSCC,null,httpsdoi.org10.7937k9tcia.2020.a8sh-7363,HNSCC-01-0001,1.3.6.1.4.1.14519.5.2.1.1706.8040.141473127645431244275904143582,RT SIMULATION,12-05-1998,NA,ADAC,RTPLAN,RT Plan Storage,1.2.840.10008.5.1.4.1.1.481.5,1,163.31 KB,./HNSCC/HNSCC-01-0001/12-05-1998-NA-RT SIMULATION-43582/1.000000-NA-27032,2021-09-30T21:38:59.207",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.710917975285844947054680280727,HNSCC,null,httpsdoi.org10.7937k9tcia.2020.a8sh-7363,HNSCC-01-0001,1.3.6.1.4.1.14519.5.2.1.1706.8040.141473127645431244275904143582,RT SIMULATION,12-05-1998,NA,ADAC,RTDOSE,RT Dose Storage,1.2.840.10008.5.1.4.1.1.481.2,1,616.53 KB,./HNSCC/HNSCC-01-0001/12-05-1998-NA-RT SIMULATION-43582/1.000000-NA-80727,2021-09-30T21:38:59.512",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.244695219130951879840699778710,HNSCC,null,httpsdoi.org10.7937k9tcia.2020.a8sh-7363,HNSCC-01-0001,1.3.6.1.4.1.14519.5.2.1.1706.8040.141473127645431244275904143582,RT SIMULATION,12-05-1998,NA,ADAC,RTSTRUCT,RT Structure Set Storage,1.2.840.10008.5.1.4.1.1.481.3,1,4.34 MB,./HNSCC/HNSCC-01-0001/12-05-1998-NA-RT SIMULATION-43582/1.000000-NA-78710,2021-09-30T21:39:00.341",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.305212762308252149853668352932,HNSCC,null,httpsdoi.org10.7937k9tcia.2020.a8sh-7363,HNSCC-01-0001,1.3.6.1.4.1.14519.5.2.1.1706.8040.227180113037748503185792359359,PETCT HEAD  NECK CA,12-01-1998,PET NO AC,GE MEDICAL SYSTEMS,PT,Positron Emission Tomography Image Storage,1.2.840.10008.5.1.4.1.1.128,93,3.62 MB,./HNSCC/HNSCC-01-0001/12-01-1998-NA-PETCT HEAD  NECK CA-59359/5.000000-PET NO AC-52932,2021-09-30T21:39:09.755",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.298197162770268400301730531515,HNSCC,null,httpsdoi.org10.7937k9tcia.2020.a8sh-7363,HNSCC-01-0001,1.3.6.1.4.1.14519.5.2.1.1706.8040.227180113037748503185792359359,PETCT HEAD  NECK CA,12-01-1998,PETCoronal,GE MEDICAL SYSTEMS,PT,Positron Emission Tomography Image Storage,1.2.840.10008.5.1.4.1.1.128,150,6.10 MB,./HNSCC/HNSCC-01-0001/12-01-1998-NA-PETCT HEAD  NECK CA-59359/601.000000-PETCoronal-31515,2021-09-30T21:39:10.428",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.868288851446304820044692604291,HNSCC,null,httpsdoi.org10.7937k9tcia.2020.a8sh-7363,HNSCC-01-0001,1.3.6.1.4.1.14519.5.2.1.1706.8040.227180113037748503185792359359,PETCT HEAD  NECK CA,12-01-1998,PET AC,GE MEDICAL SYSTEMS,PT,Positron Emission Tomography Image Storage,1.2.840.10008.5.1.4.1.1.128,93,3.62 MB,./HNSCC/HNSCC-01-0001/12-01-1998-NA-PETCT HEAD  NECK CA-59359/4.000000-PET AC-04291,2021-09-30T21:39:12.665",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.185781957936988553557006908612,HNSCC,null,httpsdoi.org10.7937k9tcia.2020.a8sh-7363,HNSCC-01-0001,1.3.6.1.4.1.14519.5.2.1.1706.8040.313496866457711298743321192442,PETCT HEAD  NECK CA,12-01-1998,PETCoronal,GE MEDICAL SYSTEMS,PT,Positron Emission Tomography Image Storage,1.2.840.10008.5.1.4.1.1.128,150,6.09 MB,./HNSCC/HNSCC-01-0001/12-01-1998-NA-PETCT HEAD  NECK CA-92442/601.000000-PETCoronal-08612,2021-09-30T21:39:16.019",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.182477636067594830828043829271,HNSCC,null,httpsdoi.org10.7937k9tcia.2020.a8sh-7363,HNSCC-01-0001,1.3.6.1.4.1.14519.5.2.1.1706.8040.313496866457711298743321192442,PETCT HEAD  NECK CA,12-01-1998,PET NO AC,GE MEDICAL SYSTEMS,PT,Positron Emission Tomography Image Storage,1.2.840.10008.5.1.4.1.1.128,223,8.66 MB,./HNSCC/HNSCC-01-0001/12-01-1998-NA-PETCT HEAD  NECK CA-92442/5.000000-PET NO AC-29271,2021-09-30T21:39:31.837",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.477256788493480202565445213575,HNSCC,null,httpsdoi.org10.7937k9tcia.2020.a8sh-7363,HNSCC-01-0001,1.3.6.1.4.1.14519.5.2.1.1706.8040.313496866457711298743321192442,PETCT HEAD  NECK CA,12-01-1998,PET AC,GE MEDICAL SYSTEMS,PT,Positron Emission Tomography Image Storage,1.2.840.10008.5.1.4.1.1.128,223,8.68 MB,./HNSCC/HNSCC-01-0001/12-01-1998-NA-PETCT HEAD  NECK CA-92442/4.000000-PET AC-13575,2021-09-30T21:39:37.173",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.157037781170457401782698627824,HNSCC,null,httpsdoi.org10.7937k9tcia.2020.a8sh-7363,HNSCC-01-0001,1.3.6.1.4.1.14519.5.2.1.1706.8040.442458148121439101275970714500,PETCT HEAD  NECK CA,03-27-1999,PETCoronal,GE MEDICAL SYSTEMS,PT,Positron Emission Tomography Image Storage,1.2.840.10008.5.1.4.1.1.128,150,6.09 MB,./HNSCC/HNSCC-01-0001/03-27-1999-NA-PETCT HEAD  NECK CA-14500/601.000000-PETCoronal-27824,2021-09-30T21:39:49.626",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.179676857865071728204685714318,HNSCC,null,httpsdoi.org10.7937k9tcia.2020.a8sh-7363,HNSCC-01-0001,1.3.6.1.4.1.14519.5.2.1.1706.8040.442458148121439101275970714500,PETCT HEAD  NECK CA,03-27-1999,PET NO AC,GE MEDICAL SYSTEMS,PT,Positron Emission Tomography Image Storage,1.2.840.10008.5.1.4.1.1.128,91,3.53 MB,./HNSCC/HNSCC-01-0001/03-27-1999-NA-PETCT HEAD  NECK CA-14500/6.000000-PET NO AC-14318,2021-09-30T21:39:52.431",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.324195958178312580902646761630,HNSCC,null,httpsdoi.org10.7937k9tcia.2020.a8sh-7363,HNSCC-01-0001,1.3.6.1.4.1.14519.5.2.1.1706.8040.442458148121439101275970714500,PETCT HEAD  NECK CA,03-27-1999,PET AC,GE MEDICAL SYSTEMS,PT,Positron Emission Tomography Image Storage,1.2.840.10008.5.1.4.1.1.128,91,3.54 MB,./HNSCC/HNSCC-01-0001/03-27-1999-NA-PETCT HEAD  NECK CA-14500/5.000000-PET AC-61630,2021-09-30T21:39:55.196",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.210007744570635546166714525068,HNSCC,null,httpsdoi.org10.7937k9tcia.2020.a8sh-7363,HNSCC-01-0001,1.3.6.1.4.1.14519.5.2.1.1706.8040.313496866457711298743321192442,PETCT HEAD  NECK CA,12-01-1998,CT Atten Cor Head IN,GE MEDICAL SYSTEMS,CT,CT Image Storage,1.2.840.10008.5.1.4.1.1.2,223,117.84 MB,./HNSCC/HNSCC-01-0001/12-01-1998-NA-PETCT HEAD  NECK CA-92442/2.000000-CT Atten Cor Head IN-25068,2021-09-30T21:40:00.133",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.291436290464093187619122791259,HNSCC,null,httpsdoi.org10.7937k9tcia.2020.a8sh-7363,HNSCC-01-0001,1.3.6.1.4.1.14519.5.2.1.1706.8040.330854768746477145019619129348,PETCT HEAD  NECK CA,03-27-1999,CT Atten Cor Head IN,GE MEDICAL SYSTEMS,CT,CT Image Storage,1.2.840.10008.5.1.4.1.1.2,223,117.84 MB,./HNSCC/HNSCC-01-0001/03-27-1999-NA-PETCT HEAD  NECK CA-29348/2.000000-CT Atten Cor Head IN-91259,2021-09-30T21:40:23.179",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.898612621408214889027419428768,HNSCC,null,httpsdoi.org10.7937k9tcia.2020.a8sh-7363,HNSCC-01-0001,1.3.6.1.4.1.14519.5.2.1.1706.8040.141473127645431244275904143582,RT SIMULATION,12-05-1998,NA,Philips Medical Systems,CT,CT Image Storage,1.2.840.10008.5.1.4.1.1.2,117,61.63 MB,./HNSCC/HNSCC-01-0001/12-05-1998-NA-RT SIMULATION-43582/NA-28768,2021-09-30T21:39:39.137"
        ]
        

        self.organ_test_data = [
            "/../test_data/HNSCC-01-0001/Left_Parotid.npy",
            "/../test_data/HNSCC-01-0001/Right_Parotid.npy",
            "/../test_data/HNSCC-01-0001/Brainstem.npy",
            "/../test_data/HNSCC-01-0001/Left_Parotid.npy",

        ]
        self.structured_test_data = self.get_data_as_dict(self.test_read_data)

    def get_data_as_dict(self, data):
        keys = self.read_data_titles.split(',')
        structured_test_data = []
        for row in data:
            row_dict = {}
            for i, value in enumerate(row.split(',')):
                row_dict[keys[i]] = value
            structured_test_data.append(row_dict)

        return structured_test_data

    def test_happy_scenario (self):
        rstruct_data_uid = "1.3.6.1.4.1.14519.5.2.1.1706.8040.244695219130951879840699778710"
        ct_image_data_uids = [
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.898612621408214889027419428768",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.210007744570635546166714525068",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.291436290464093187619122791259"
        ]
        for row in self.structured_test_data:
            root_dir = os.getcwd().replace("models", "")
            series = Series(root_dir, row)
            self.assertRegex(series.series_uid, "1.3.6.1.4.1.14519.5.2.1.1706.8040.\\d{30}")
            self.assertEqual(series.subject_ID, "HNSCC-01-0001")
            self.assertEqual(series.collection, "HNSCC")
            self.assertGreater(len(series.study_description), 0)
            self.assertRegex(series.study_date, "\\d\\d-\\d\\d-\\d\\d\\d\\d")
            self.assertGreater(len(series.modality), 0)
            if series.series_uid in ct_image_data_uids:
                # self.assertGreater(int(series.number_of_images), 0)
                self.evaluate_ct_data(series.medical_images)
            else:
                # self.assertLessEqual(int(series.number_of_images), 1)
                self.assertEqual(len(series.medical_images), 0)
            self.assertRegex(series.file_location, "./HNSCC/HNSCC-01-0001/.+")
            self.assertEqual(series.root_directory, root_dir)
            if series.series_uid == rstruct_data_uid:
                self.evaluate_contour_data(series.contours_data)
            else:
                self.assertEqual(len(series.contours_data), 0)
                self.assertEqual(len(series.organs), 0)
            # self.assertEqual(series.organs, "HNSCC")

    def evaluate_contour_data(self, contour_data):
        self.assertEqual(len(contour_data), 6)
        included_contours = [
            'marked isocenter', 'brainstem', 'rt parotid', 'lt parotid', 'external', 'ring'
        ]
        self.assertCountEqual(contour_data.keys(), included_contours)
        for value in contour_data.values():
            self.assertGreater(len(value), 0)

    def evaluate_ct_data(self, ct_data):
        self.assertGreater(len(ct_data), 0)

        for ct_image in ct_data:
            self.assertTrue(hasattr(ct_image, "dicomparser"))
            self.assertTrue(hasattr(ct_image, "pydicom"))
            self.assertTrue(hasattr(ct_image, "zlocation"))
