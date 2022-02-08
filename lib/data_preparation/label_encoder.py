#
# Main
#

class LabelEncoder:
    classes = [
        "paved",
        "asphalt",
        "concrete",
        "concrete lanes",
        "concrete plates",
        "paving stones",
        "sett",
        "unhewn cobblestone",
        "cobblestone",
        "metal",
        "wood",
        "stepping_stones",
        "unpaved",
        "compacted",
        "fine gravel",
        "gravel",
        "rock",
        "pebblestone",
        "ground",
        "dirt",
        "earth",
        "grass",
        "mud",
        "sand",
        "woodchips",
        "snow",
        "ice",
        "salt",
    ]

    def num_classes(self):
        return len(self.classes)

    def label_to_index(self, label):
        return self.classes.index(label)

    def index_to_label(self, index: int):
        return self.classes[index]
