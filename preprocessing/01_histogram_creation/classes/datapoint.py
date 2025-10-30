# Class Datapoint
# Combines the variable name, the value of the variable and the timestamp of the measurement into one object
# Author: Lukas L. Köning
from enum import Enum


class RoadType(Enum):
    UNKNOWN = 0
    CITY = 1
    COUNTY = 2
    HIGHWAY = 3
    COMBINED = 4


class Datapoint:
    def __init__(self, variable_name, variable_value, time_of_measurement, road_type=RoadType.UNKNOWN):
        self.varname = variable_name
        self.value = variable_value
        self.timestamp = time_of_measurement
        self.road_type = road_type

    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other

    def __eq__(self, other):
        return self.value == other

    def __sub__(self, other):
        if type(other) is not Datapoint:
            raise TypeError("Can only subtract two Datapoints.")
        return Datapoint(variable_name=self.varname,
                         variable_value=float(self.value) - float(other.value),
                         time_of_measurement=self.timestamp,
                         road_type=self.road_type)

    def set_road_type(self, road_type: RoadType):
        """
        Update the road type.
        :param road_type:
        :return:
        """
        self.road_type = road_type
