# Class DataType
# Combines relevant information about a data type
# Author: Lukas L. Köning

class DataType:
    def __init__(self, id, variable_name, name_de, value_unit):
        self.id = id
        self.var_name = variable_name
        self.german_name = name_de
        self.unit = value_unit

    def __eq__(self, other):
        return self.id == other.id and self.var_name == other.var_name and self.german_name == other.german_name \
            and self.unit == other.unit
