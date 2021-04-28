from enum import Enum


class MMDetectionTaskType(Enum):
    OBJECTDETECTION = 0
    INSTANCESEGMENTATION = 1

    def __str__(self):
        return self.name
