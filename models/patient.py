class Patient:
    def __init__(self, row):
        self.subject_ID = row.get("Subject ID")
        self.series = []