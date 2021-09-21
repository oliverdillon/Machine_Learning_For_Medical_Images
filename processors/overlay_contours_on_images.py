class Overlay_contours_on_images:
    def __init__(self,dataset):
        self.dataset = dataset
        self.filter_patients_by_modality()

    def is_required_data(self,series):
        if series.modality =="CT":
            return True
        elif series.modality =="RTSTRUCT":
            return True
        return False

    def filter_patients_by_modality(self):
        data = self.dataset.data
        for patient in data:
            series_filtered = []
            series_iterator = filter(self.is_required_data,patient.series)
            for series in series_iterator:
                series_filtered.append(series)
            patient.series = series_filtered
