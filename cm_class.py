class CMThreshold:
    def __init__(self, threshold, cm):
        self.threshold = threshold
        self.cm = cm
    def display_info(self):
        print(f"Threshold: {self.threshold}")
        print(f"Confusion Matrix: {self.cm}")
        
