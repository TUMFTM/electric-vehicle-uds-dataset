# Class BEVUsage
# Parent class for classes Track and Charging
# Author: Lukas L. Köning
import calendar
import logging
import multiprocessing as mp
import os
import time
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Union, List

import numpy as np
import psycopg2

from config import THREADS
from .datapoint import Datapoint, RoadType
from .datatype import DataType
from .exceptions import RawDataError
from .histogram import Histogram, Histogram2D, Bins


class HistogramType(Enum):
    # Track Histogram types
    CELL_C_RATE = 0
    PACK_TEMP_MAX = 1
    PACK_TEMP_MIN = 2
    PACK_TEMP_DELTA = 3
    PACK_SOC = 4
    PACK_DOD = 5
    VEHICLE_SPEED = 6
    TRACK_DURATION = 7
    IDLE_PERIOD_DURATION = 8
    PACK_VOLTAGE = 9
    CELL_VOLTAGE_MAX = 10
    CELL_VOLTAGE_MIN = 11
    CELL_VOLTAGE_DELTA = 12
    AMBIENT_AIR_TEMP = 13
    COOLANT_TEMP_INVERTER_INLET = 14
    PACK_COOLANT_TEMP_INLET = 15
    PACK_COOLANT_TEMP_OUTLET = 16
    HEATPUMP_POWER = 17
    ROTOR_REAR_TEMP = 18
    STATOR_REAR_TEMP = 19
    INTERIOR_TEMP = 20
    PTC1_CURRENT = 21
    PTC2_CURRENT = 22
    PTC_VOLTAGE = 23
    REAR_INVERTER_TEMP = 24
    TRAVELED_DISTANCE = 25
    AUXILIARIES_POWER = 26
    PTC1_POWER = 27
    PTC2_POWER = 28
    C_RATE_PEAK_ANALYSIS_AMPL = 29
    C_RATE_PEAK_ANALYSIS_FREQ = 30

    HIST2D_C_RATE__PACK_TEMP_MAX = 50
    HIST2D_C_RATE__PACK_TEMP_DELTA = 51
    HIST2D_C_RATE__PACK_SOC = 52
    HIST2D_C_RATE__VEHICLE_SPEED = 53
    HIST2D_PACK_TEMP_MAX__PACK_SOC = 54

    # Charging histogram types
    CHG_POWER = 100
    CHG_HIST2D_POWER__PACK_TEMP_MAX = 101
    CHG_HIST2D_POWER__PACK_TEMP_DELTA = 102
    CHG_DURATION = 103
    CHG_HIST2D_POWER__SOC = 104
    CHG_EOC = 105


class BEVUsage:
    BINS_FOR_HISTOGRAMTYPE = {
        HistogramType.CELL_C_RATE: (Bins(minimum=-16, maximum=16, step_size=0.01), None),
        HistogramType.PACK_TEMP_MAX: (Bins(minimum=-20, maximum=50, step_size=0.25), None),
        HistogramType.PACK_TEMP_MIN: (Bins(minimum=-20, maximum=50, step_size=0.25), None),
        HistogramType.PACK_TEMP_DELTA: (Bins(minimum=0, maximum=10, step_size=0.05), None),
        HistogramType.PACK_SOC: (Bins(minimum=-0.2, maximum=100.3, step_size=0.4), None),
        HistogramType.PACK_DOD: (Bins(minimum=0.2, maximum=100.3, step_size=0.4), None),
        HistogramType.VEHICLE_SPEED: (Bins(minimum=0, maximum=250, step_size=1), None),
        HistogramType.TRACK_DURATION: (Bins(minimum=0, maximum=480, step_size=1), None),
        HistogramType.IDLE_PERIOD_DURATION: (Bins(minimum=0, maximum=20160, step_size=15), None),
        HistogramType.PACK_VOLTAGE: (Bins(minimum=0, maximum=500, step_size=1), None),
        HistogramType.CELL_VOLTAGE_MAX: (Bins(minimum=2, maximum=5, step_size=0.01), None),
        HistogramType.CELL_VOLTAGE_MIN: (Bins(minimum=2, maximum=5, step_size=0.01), None),
        HistogramType.CELL_VOLTAGE_DELTA: (Bins(minimum=0, maximum=1, step_size=0.005), None),
        HistogramType.CHG_POWER: (Bins(minimum=0, maximum=400, step_size=0.1), None),
        HistogramType.CHG_DURATION: (Bins(minimum=0, maximum=720, step_size=1), None),
        HistogramType.CHG_EOC: (Bins(minimum=0, maximum=100, step_size=1), None),
        HistogramType.AMBIENT_AIR_TEMP: (Bins(minimum=-20, maximum=80, step_size=0.25), None),
        HistogramType.COOLANT_TEMP_INVERTER_INLET: (Bins(minimum=0, maximum=100, step_size=0.25), None),
        HistogramType.PACK_COOLANT_TEMP_INLET: (Bins(minimum=0, maximum=100, step_size=0.25), None),
        HistogramType.PACK_COOLANT_TEMP_OUTLET: (Bins(minimum=0, maximum=100, step_size=0.25), None),
        HistogramType.HEATPUMP_POWER: (Bins(minimum=0, maximum=20000, step_size=1), None),
        HistogramType.ROTOR_REAR_TEMP: (Bins(minimum=-20, maximum=120, step_size=0.25), None),
        HistogramType.STATOR_REAR_TEMP: (Bins(minimum=-20, maximum=120, step_size=0.25), None),
        HistogramType.INTERIOR_TEMP: (Bins(minimum=-20, maximum=100, step_size=0.25), None),
        HistogramType.PTC1_CURRENT: (Bins(minimum=0, maximum=100, step_size=0.1), None),
        HistogramType.PTC2_CURRENT: (Bins(minimum=0, maximum=100, step_size=0.1), None),
        HistogramType.PTC_VOLTAGE: (Bins(minimum=0, maximum=500, step_size=1), None),
        HistogramType.REAR_INVERTER_TEMP: (Bins(minimum=-20, maximum=200, step_size=0.25), None),
        HistogramType.TRAVELED_DISTANCE: (Bins(minimum=0, maximum=1000, step_size=1), None),
        HistogramType.AUXILIARIES_POWER: (Bins(minimum=0, maximum=20000, step_size=1), None),
        HistogramType.PTC1_POWER: (Bins(minimum=0, maximum=10000, step_size=1), None),
        HistogramType.PTC2_POWER: (Bins(minimum=0, maximum=10000, step_size=1), None),
        HistogramType.C_RATE_PEAK_ANALYSIS_AMPL: (Bins(minimum=0, maximum=16, step_size=0.01), None),
        HistogramType.C_RATE_PEAK_ANALYSIS_FREQ: (Bins(minimum=0, maximum=20, step_size=0.01), None)
    }

    TITLE_FOR_HISTOGRAMTYPE = {
        HistogramType.CELL_C_RATE: "C-Rate der Batteriezelle",
        HistogramType.PACK_TEMP_MAX: "Maximale Temperatur des Batteriepacks",
        HistogramType.PACK_TEMP_MIN: "Minimale Temperatur des Batteriepacks",
        HistogramType.PACK_TEMP_DELTA: "Maximale Temperaturdifferenz im Batteriepack",
        HistogramType.PACK_SOC: "State-Of-Charge (SOC) der Hochvoltbatterie",
        HistogramType.PACK_DOD: "Depth-Of-Discharge (DOD) der Hochvoltbatterie pro Fahrt",
        HistogramType.VEHICLE_SPEED: "Fahrzeuggeschwindigkeit",
        HistogramType.TRACK_DURATION: "Fahrtdauer",
        HistogramType.IDLE_PERIOD_DURATION: "Ruhezeit",
        HistogramType.PACK_VOLTAGE: "Spannung des Batteriepacks",
        HistogramType.CELL_VOLTAGE_MAX: "Maximale Einzelzellspannung des Batteriepacks",
        HistogramType.CELL_VOLTAGE_MIN: "Minimale Einzelzellspannung des Batteriepacks",
        HistogramType.CELL_VOLTAGE_DELTA: "Maximale Einzelzellspannungsdifferenz im Batteriepack",
        HistogramType.CHG_POWER: "Ladeleistung",
        HistogramType.CHG_DURATION: "Ladedauer",
        HistogramType.CHG_EOC: "Ladehub (EOC) der Hochvoltbatterie pro Ladung",
        HistogramType.HIST2D_C_RATE__PACK_TEMP_MAX: "2D Histogramm der C-Rate über die maximale Packtemperatur",
        HistogramType.HIST2D_C_RATE__PACK_TEMP_DELTA: "2D Histogramm der C-Rate über die maximale Packtemperaturdifferenz",
        HistogramType.HIST2D_C_RATE__PACK_SOC: "2D Histogramm der C-Rate über den SOC",
        HistogramType.HIST2D_C_RATE__VEHICLE_SPEED: "2D Histogramm der C-Rate über die Fahrzeuggeschwindigkeit",
        HistogramType.HIST2D_PACK_TEMP_MAX__PACK_SOC: "2D Histogramm der maximalen Packtemperatur über den SOC",
        HistogramType.CHG_HIST2D_POWER__PACK_TEMP_MAX: "2D Histogramm der Ladeleistung über die maximalen Packtemperatur",
        HistogramType.CHG_HIST2D_POWER__PACK_TEMP_DELTA: "2D Histogramm der Ladeleistung über die maximale Packtemperaturdifferenz",
        HistogramType.CHG_HIST2D_POWER__SOC: "2D Histogramm der Ladeleistung über den SOC",
        HistogramType.AMBIENT_AIR_TEMP: "Umgebungstemperatur",
        HistogramType.COOLANT_TEMP_INVERTER_INLET: "Temperatur der Kühlflüssigkeit am Einlass Wechselrichter",
        HistogramType.PACK_COOLANT_TEMP_INLET: "Temperatur der Kühlflüssigkeit am Einlass Batteriepack",
        HistogramType.PACK_COOLANT_TEMP_OUTLET: "Temperatur der Kühlflüssigkeit am Auslass Batteriepack",
        HistogramType.HEATPUMP_POWER: "Leistungsaufnahme der Wärmepumpe",
        HistogramType.ROTOR_REAR_TEMP: "Temperatur des Rotors des Hinterachsmotors",
        HistogramType.STATOR_REAR_TEMP: "Temperatur des Stators des Hinterachsmotors",
        HistogramType.INTERIOR_TEMP: "Innenraumtemperatur",
        HistogramType.PTC1_CURRENT: "PTC 1 Strom",
        HistogramType.PTC2_CURRENT: "PTC 2 Strom",
        HistogramType.PTC_VOLTAGE: "PTC Spannung",
        HistogramType.REAR_INVERTER_TEMP: "Temperatur des Inverters an der Hinterachse",
        HistogramType.TRAVELED_DISTANCE: "Zurückgelegte Strecke pro Fahrt",
        HistogramType.AUXILIARIES_POWER: "Leistungsaufnahme der Nebenverbraucher",
        HistogramType.PTC1_POWER: "PTC 1 Leistung",
        HistogramType.PTC2_POWER: "PTC 2 Leistung",
        HistogramType.C_RATE_PEAK_ANALYSIS_AMPL: "Amplituden der C-Raten Peaks",
        HistogramType.C_RATE_PEAK_ANALYSIS_FREQ: "Frequenzen der C-Raten Peaks"
    }

    def __init__(self, database_id, start, end, vehicle, processed, prior_usage_end=None):
        self.id = database_id
        self.vehicle = vehicle
        self.start_time = start
        self.end_time = end
        self.unprocessed = not processed
        self.prior_usage_end = prior_usage_end

        # Initialize some variables
        self.value_id_dict = {}
        self.can_data = {}
        self.histogram_list = []
        self.cache = {}

        # Setup bins for 2D histograms
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.HIST2D_C_RATE__PACK_TEMP_MAX] = (
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.CELL_C_RATE][0],
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.PACK_TEMP_MAX][0])
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.HIST2D_C_RATE__PACK_TEMP_DELTA] = (
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.CELL_C_RATE][0],
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.PACK_TEMP_DELTA][0])
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.HIST2D_C_RATE__PACK_SOC] = (
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.CELL_C_RATE][0],
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.PACK_SOC][0])
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.HIST2D_C_RATE__VEHICLE_SPEED] = (
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.CELL_C_RATE][0],
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.VEHICLE_SPEED][0])
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.HIST2D_PACK_TEMP_MAX__PACK_SOC] = (
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.PACK_TEMP_MAX][0],
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.PACK_SOC][0])
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.CHG_HIST2D_POWER__PACK_TEMP_MAX] = (
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.CHG_POWER][0],
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.PACK_TEMP_MAX][0])
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.CHG_HIST2D_POWER__PACK_TEMP_DELTA] = (
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.CHG_POWER][0],
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.PACK_TEMP_DELTA][0])
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.CHG_HIST2D_POWER__SOC] = (
        self.BINS_FOR_HISTOGRAMTYPE[HistogramType.CHG_POWER][0], self.BINS_FOR_HISTOGRAMTYPE[HistogramType.PACK_SOC][0])

    # Virtual methods
    def analyze(self, desired_histograms: List[HistogramType], plot=False):
        """
        This function analyzes the bev usage and creates histograms.
        :param desired_histograms: List of HistogramTypes of which histograms should be calculated
        :param plot: Bool, plot the calculated histograms?
        :return:
        """
        raise NotImplementedError()

    def save_histograms_to_database(self, force_overwrite=False):
        """
        Saves the calculated Histograms from the internal list self.histograms_list to the database
        :return:
        """
        raise NotImplementedError()

    def calculate_histogram(self, histogram_type: HistogramType, plot=False, keep_cache=False):
        """
        Calculates the histogram for the given histogram type and add it to the internal list
        :param histogram_type: HistogramType, type of the histogram that should be created
        :param plot: Bool, plot the histogram?
        :param keep_cache: Bool, keep the cache or delete it?
        :return: nothing
        """
        raise NotImplementedError()

    # Implemented methods
    def get_value_id_data(self):
        """
        Fetches the data corresponding to value ids.
        :return:
        """
        conn = psycopg2.connect(database="mobtrack",
                                user=os.environ["db_user"],
                                password=os.environ["db_pwd"],
                                host=os.environ["host"],
                                port="5432")
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute('''SELECT value_id, variable_name, name_de, unit FROM vehicle.can_value''')

        sql_result = cur.fetchall()
        for value_data in sql_result:
            self.value_id_dict[value_data[0]] = DataType(value_data[0], value_data[1], value_data[2], value_data[3])

        conn.close()

    def fetch_can_data(self):
        """
        Fetches the collected can data from the database and store it as dictionary of data types with their Datapoints
        :return: The dictionary
        """
        # Verify that all needed data is available
        if len(self.value_id_dict.keys()) == 0:
            self.get_value_id_data()

        can_data_list = []
        conn = psycopg2.connect(database="mobtrack",
                                user=os.environ["db_user"],
                                password=os.environ["db_pwd"],
                                host=os.environ["host"],
                                port="5432")
        conn.autocommit = True
        cur = conn.cursor()

        sql = "SELECT can.time, cr.value_id, can.value FROM sensor.can can " \
              "INNER JOIN vehicle.can_request cr ON can.request_id = cr.request_id " \
              "WHERE can.vehicle_id=%s AND can.time BETWEEN %s AND %s;"
        sql_parameter = [self.vehicle.id, self.start_time, self.end_time]
        cur.execute(sql, sql_parameter)

        sql_result = cur.fetchall()
        for sql_can_data in sql_result:
            if sql_can_data[1] is None:
                continue
            data_name = self.value_id_dict[sql_can_data[1]].var_name
            if type(sql_can_data[2]) is Decimal:
                can_data_list.append(Datapoint(data_name, float(sql_can_data[2]), sql_can_data[0]))
            else:
                can_data_list.append(Datapoint(data_name, sql_can_data[2], sql_can_data[0]))

        cur.close()
        conn.close()

        # TODO: Use map-matching to classify each data point

        # Convert can data list to dictionary for better data access
        for dp in can_data_list:
            if dp.varname not in self.can_data.keys():
                self.can_data[dp.varname] = []
            self.can_data[dp.varname].append(dp)

        # Sort can data chronologically
        for data_type in self.can_data:
            self.can_data[data_type].sort(key=lambda x: x.timestamp, reverse=False)

        return self.can_data

    def is_complete(self):
        """
        Checks, if all sensor data is uploaded to the database and available for processing.
        :return: Bool, if all sensor data is available
        """
        # Make sure, all data is fetched from the database
        if len(self.value_id_dict.keys()) == 0:
            self.get_value_id_data()

        if len(self.can_data.keys()) == 0:
            self.fetch_can_data()

        # Check if there is some can data
        if len(self.can_data.keys()) == 0:
            return False

        # Check if sensor data is missing
        # --> Current data is mandatory
        if "hv_battery_current" not in self.can_data.keys():
            return False

        # --> Determine the measurement frequency
        pack_current = self.can_data["hv_battery_current"]
        datetime_delta_ts = [pack_current[i + 1].timestamp - pack_current[i].timestamp
                             for i in range(len(pack_current) - 1)]
        delta_ts = [datetime_delta_t.seconds + datetime_delta_t.microseconds / 1000000
                    for datetime_delta_t in datetime_delta_ts]
        measurement_frequency = self._round_to_significant_digits(float(1 / np.median(delta_ts)), 2)

        # --> Check if enough data points exists
        track_length = datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(self.start_time,
                                                                                                 "%Y-%m-%d %H:%M:%S")
        ref_num_datapoints = np.ceil(track_length.seconds * measurement_frequency)
        delta_datapoints = ref_num_datapoints - len(pack_current)

        # Epsilon is defined very conservative to prevent processing of an incomplete bev_usage
        if delta_datapoints <= 1 * 60 * measurement_frequency:
            return True
        else:
            if (datetime.now() - datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")).days > 14:
                # Assuming, that after 14 days all data should be transferred and missing data points are data gaps
                return True
            else:
                # Assuming, that the upload of data is still pending/running...
                logging.warning("--> Missing data, but track is not older than 14 days. Try processing later again!")
                return False

    def _create_histogram(self, data_type_name, bins: Union[int, Bins] = 10, road_type: RoadType = RoadType.UNKNOWN,
                          bin_digits=0):
        """
        Creates a histogram for the given datatype from the can data.
        :param data_type_name: type of the desired data ("variable_name" in vehicle.can_value)
        :param bins: Count of bins of the histogram or Bins object
        :param road_type: RoadType enum value to filter data for city, count and highway datapoints.
                          If unknown, no filter is applied.
        :param bin_digits: int, number of digits to which the start and end of bins is rounded,
                           if the bins are calculated automatically
        :return: Histogram object
        """
        # Make sure, all data is available
        if len(self.value_id_dict.keys()) == 0:
            self.get_value_id_data()

        if len(self.can_data.keys()) == 0:
            self.fetch_can_data()

        # Find the can_value object for the given datatype
        data_type = None
        for dt in self.value_id_dict.values():
            if dt.var_name == data_type_name:
                data_type = dt
                break

        if data_type is None:
            raise KeyError("The data_type '" + data_type_name + "' is not defined in the database! Check the spelling "
                                                                "or add it to the database.")

        # Create histogram
        data_for_hist = [dp for dp in self.can_data[data_type.var_name]]

        if road_type is not RoadType.UNKNOWN:
            # TODO: Add filtering of road type here!
            pass

        histogram = Histogram(data_type, road_type)
        histogram.calculate(data_for_hist, bins, bin_digits)
        return histogram

    def _create_custom_histogram(self, data_type: DataType, data, bins: Union[int, Bins],
                                 road_type: RoadType = RoadType.UNKNOWN, bin_digits=0):
        """
        Creates histogram from given data.
        :param data_type: DataType object with information about the supplied data
        :param data: List of Datapoint objects for the histogram
        :param bins: Count of bins of the histogram or Bins object
        :param road_type: RoadType enum value to filter data for city, count and highway datapoints.
                          If unknown, no filter is applied.
        :param bin_digits: int, number of digits to which the start and end of bins is rounded,
                           if the bins are calculated automatically
        :return: Histogram object
        """
        # Make sure, all data is available
        if len(self.value_id_dict.keys()) == 0:
            self.get_value_id_data()

        if len(self.can_data.keys()) == 0:
            self.fetch_can_data()

        if road_type is not RoadType.UNKNOWN:
            # TODO: Add filtering of road type here!
            pass

        histogram = Histogram(data_type, road_type)
        histogram.calculate(data, bins, bin_digits)
        return histogram

    def _create_2d_histogram(self, x_data_type_name, y_data_type_name, x_bins: Union[int, Bins],
                             y_bins: Union[int, list], road_type: RoadType = RoadType.UNKNOWN, bin_digits=(0, 0)):
        """
        Creates a 2D histogram for the given data_type
        :param x_data_type_name: type of the desired data on the x-axis ("variable_name" in vehicle.can_value)
        :param y_data_type_name: type of the desired data on the y-axis ("variable_name" in vehicle.can_value)
        :param x_bins: Count of bins of the histogram or Bins object
        :param y_bins: Count of bins of the histogram or Bins object
        :param road_type: RoadType enum value to filter data for city, count and highway datapoints.
                          If unknown, no filter is applied.
        :param bin_digits: tuple, number of digits to which the start and end of bins is rounded,
                           if the bins are calculated automatically
        :return: Histogram2D object
        """
        # Make sure, all data is available
        if len(self.value_id_dict.keys()) == 0:
            self.get_value_id_data()

        if len(self.can_data.keys()) == 0:
            self.fetch_can_data()

        # Find the can_value object for the given datatype
        x_data_type = None
        y_data_type = None
        for dt in self.value_id_dict.values():
            if dt.var_name == x_data_type_name:
                x_data_type = dt
            elif dt.var_name == y_data_type_name:
                y_data_type = dt

            if x_data_type is not None and y_data_type is not None:
                break

        if x_data_type is None:
            raise KeyError("The data_type '" + x_data_type_name + "' is not defined in the database! Check the spelling"
                                                                  " or add it to the database.")
        if y_data_type is None:
            raise KeyError("The data_type '" + y_data_type_name + "' is not defined in the database! Check the spelling"
                                                                  " or add it to the database.")

        # Calculate histogram
        data_x = [dp for dp in self.can_data[x_data_type.var_name]]
        data_y = [dp for dp in self.can_data[y_data_type.var_name]]
        hist2d = Histogram2D(x_data_type=x_data_type, y_data_type=y_data_type, road_type=road_type)
        hist2d.calculate(data_x=data_x, data_y=data_y, bins_x=x_bins, bins_y=y_bins, bin_digits=bin_digits)
        return hist2d

    def _create_custom_2d_histogram(self, x_data_type, y_data_type, x_bins: Union[int, Bins], y_bins: Union[int, list],
                                    x_data, y_data, road_type: RoadType = RoadType.UNKNOWN, bin_digits=(0, 0)):
        """
        Creates a 2D histogram for the given data_type
        :param x_data_type: DataType object describing the x axis data
        :param y_data_type: DataType object describing the y axis data
        :param x_bins: Count of bins of the histogram or Bins object
        :param y_bins: Count of bins of the histogram or Bins object
        :param x_data: List of Datapoint objects for the x axis of the histogram
        :param y_data: List of Datapoint objects for the y axis of the histogram
        :param road_type: RoadType enum value to filter data for city, count and highway datapoints.
                          If unknown, no filter is applied.
        :param bin_digits: tuple, number of digits to which the start and end of bins is rounded,
                           if the bins are calculated automatically
        :return: Histogram2D object
        """
        # Make sure, all data is available
        if len(self.value_id_dict.keys()) == 0:
            self.get_value_id_data()

        if len(self.can_data.keys()) == 0:
            self.fetch_can_data()

        hist2d = Histogram2D(x_data_type=x_data_type, y_data_type=y_data_type, road_type=road_type)
        hist2d.calculate(data_x=x_data, data_y=y_data, bins_x=x_bins, bins_y=y_bins, bin_digits=bin_digits)
        return hist2d

    def _check_if_already_processed(self, histogram_types):
        """
        Checks, if the requested histograms are already processed.
        :param histogram_types: List, HistogramTypes of histograms that should be calculated.
        :return: List of HistogramTypes, that still need to be calculated.
        """
        raise NotImplementedError()

    # static methods
    @staticmethod
    def _array_to_sql_string_array(array):
        """
        Converts an array to a sql readable string
        :param array:
        :return:
        """
        if type(array) == list:
            array = np.array(array)
        return_string = "'{"
        for i in array:
            if len(array.shape) == 1:
                return_string += str(i) + ", "
            elif len(array.shape) == 2:
                return_string += "{"
                for j in i:
                    return_string += str(j) + ", "
                return_string = return_string[:-2] + "}, "
        if return_string[-1] == " ":
            return_string = return_string[:-2] + "}'"
        else:
            return_string += "}'"
        return return_string

    @staticmethod
    def _get_key_for_value(dictionary, value):
        """
        Returns the first key that matches the value in the dictionary
        :param dictionary: Dictionary
        :param value: value for which the key should be determined
        :return: first key that has value as its value in the dictionary
        """
        return [k for k, v in dictionary.items() if v == value][0]

    @staticmethod
    def _round_to_significant_digits(value: float, digits=2):
        """
        Rounds the given value to the next bigger absolute value with a defined number of significant digits,
        if it is smaller than |1|. Otherwise, it rounds the given value to the next bigger absolute value while keeping
        the defined number of digits.
        :param value: float, value to be rounded
        :param digits: number of significant digits
        :return: rounded float
        """
        if value > 0:
            if value >= 1:
                return np.ceil(value * np.power(10, digits)) / np.power(10, digits)
            else:
                # Minus one for correct meaning of digits
                sign_digits = (digits - 1) - int(np.floor(np.log10(abs(value))))
                return np.ceil(value * np.power(10, sign_digits)) / np.power(10, sign_digits)
        elif value < 0:
            if value <= -1:
                return np.floor(value * np.power(10, digits)) / np.power(10, digits)
            else:
                # Minus one for correct meaning of digits
                sign_digits = (digits - 1) - int(np.floor(np.log10(abs(value))))
                return np.floor(value * np.power(10, sign_digits)) / np.power(10, sign_digits)
        else:
            return 0

    @staticmethod
    def _match_data_to_timeseries(data, timeseries):
        """
        Calculates the corresponding values of the date for each timestamp in the timeseries
        :param data: List of Datapoints, sorted by timestamp
        :param timeseries: List of datetimes, must be within the data timestamps
        :return: List with values that match the timestamps in timeseries
        """
        sorted(data, key=lambda x: x.timestamp, reverse=False)
        if data[0].timestamp > timeseries[0] or data[-1].timestamp < timeseries[-1]:
            raise RawDataError("The given timeseries contains timestamps outside the data!")

        matched_data = []
        prev_i = 0
        for timestamp in timeseries:
            size_before = len(matched_data)
            found = False
            for i in range(prev_i, len(data) - 1):
                dp_i = data[i]
                dp_i1 = data[i + 1]
                if dp_i.timestamp <= timestamp < dp_i1.timestamp:
                    found = True
                    prev_i = i
                    # If the value keeps constant
                    if dp_i.value == dp_i1.value:
                        matched_data.append(Datapoint(dp_i.varname, dp_i.value, timestamp, dp_i.road_type))
                    # Else interpolate linearly between the two datapoints
                    else:
                        interpolated_value = np.interp([BEVUsage._datetime_to_timestamp(timestamp)],
                                                       [BEVUsage._datetime_to_timestamp(dp_i.timestamp),
                                                        BEVUsage._datetime_to_timestamp(dp_i1.timestamp)],
                                                       [float(dp_i.value), float(dp_i1.value)])
                        matched_data.append(Datapoint(dp_i.varname, interpolated_value, timestamp, dp_i.road_type))
                    break
            if not found:
                for i in range(len(data) - 1):
                    dp_i = data[i]
                    dp_i1 = data[i + 1]
                    if dp_i.timestamp <= timestamp < dp_i1.timestamp:
                        prev_i = i
                        # If the value keeps constant
                        if dp_i.value == dp_i1.value:
                            matched_data.append(Datapoint(dp_i.varname, dp_i.value, timestamp, dp_i.road_type))
                        # Else interpolate linearly between the two datapoints
                        else:
                            interpolated_value = np.interp([BEVUsage._datetime_to_timestamp(timestamp)],
                                                           [BEVUsage._datetime_to_timestamp(dp_i.timestamp),
                                                            BEVUsage._datetime_to_timestamp(dp_i1.timestamp)],
                                                           [float(dp_i.value), float(dp_i1.value)])
                            matched_data.append(
                                Datapoint(dp_i.varname, interpolated_value, timestamp, dp_i.road_type))
                        break
            if len(matched_data) == size_before:
                raise ValueError("Could not calculate value for timestamp!")

        return matched_data

    @staticmethod
    def _add_data_gaps_to_timeseries(example_data, timeseries):
        """
        Removes timestamps from timeseries, where data gaps exist.
        :param example_data: List of datapoints, with a measurement frequency higher than the timeseries frequency
        :param timeseries: List of datetime objects
        :return: cropped timeseries
        """
        # Determine the timeseries delta time
        timeseries_delta_t_timedelta = timeseries[1] - timeseries[0]
        timeseries_delta_t = timeseries_delta_t_timedelta.seconds + timeseries_delta_t_timedelta.microseconds / 1e6

        # Determine the data delta times
        datetime_delta_ts = [example_data[i + 1].timestamp - example_data[i].timestamp
                             for i in range(len(example_data) - 1)]
        delta_ts = [datetime_delta_t.seconds + datetime_delta_t.microseconds / 1000000
                    for datetime_delta_t in datetime_delta_ts]

        if len(timeseries) > 10000:
            # Create batches
            batches = []
            end = 0
            while end < len(timeseries):
                valid_cutoff = False
                if end + 10000 < len(timeseries):
                    end_timestamp = timeseries[end + 10000]
                else:
                    # end_timestamp = timeseries[-1]
                    batches.append(timeseries[end:-1])
                    break
                for i in range(len(example_data) - 1):
                    dp_i = example_data[i]
                    dp_i1 = example_data[i + 1]
                    if dp_i.timestamp <= end_timestamp < dp_i1.timestamp:
                        delta_t = delta_ts[i]
                        if delta_t <= timeseries_delta_t:
                            valid_cutoff = True
                            break
                if valid_cutoff:
                    batches.append(timeseries[end:end + 10000])
                    end = end + 10001
                else:
                    if end > (len(batches) + 1) * 10000 + 2500:
                        # No valid cutoff point could be found in reasonable distance to desired location
                        break
                    else:
                        end = end + 10

            # No batches found because no valid cutoff point found
            # Trying to find first valid cutoff point and process only first batch.
            if len(batches) == 0:
                end = 0
                while end < len(timeseries):
                    valid_cutoff = False
                    if end + 10000 < len(timeseries):
                        end_timestamp = timeseries[end + 10000]
                    else:
                        end_timestamp = timeseries[-1]
                    for i in range(len(example_data) - 1):
                        dp_i = example_data[i]
                        dp_i1 = example_data[i + 1]
                        if dp_i.timestamp <= end_timestamp < dp_i1.timestamp:
                            delta_t = delta_ts[i]
                            if delta_t <= timeseries_delta_t:
                                valid_cutoff = True
                                break
                    if valid_cutoff:
                        batches.append(timeseries[0:end + 10000])
                        break
                    else:
                        end = end - 10

                # If still no valid cutoff point found, data must be checked!
                if len(batches) == 0:
                    raise ValueError("Can't find valid cutoff point for batchwise calculations!")

            # Process batches parallelized
            pool = mp.Pool(THREADS)
            logging.info("--> Cropping timeseries ({} batches asynchronously)...".format(len(batches)))
            start_time = time.time()
            result_objects = [
                pool.apply_async(BEVUsage._crop_timeseries, args=(i, batch, example_data, delta_ts, timeseries_delta_t))
                for i, batch in enumerate(batches)]

            results = []
            for r in result_objects:
                results.append((r.get()[0], r.get()[1]))

            pool.close()
            pool.join()

            results.sort(key=lambda x: x[0])
            cropped_timeseries = []
            for r in results:
                cropped_timeseries.extend(r[1])

        else:
            logging.info("--> Cropping timeseries...")
            start_time = time.time()
            _, cropped_timeseries = BEVUsage._crop_timeseries(0, timeseries, example_data, delta_ts, timeseries_delta_t)

        end_time = time.time()
        dt = end_time - start_time
        logging.info("Finished cropping. It took {} hours, {} minutes and {} seconds.".format(round(dt / 3600),
                                                                                              round((dt % 3600) / 60),
                                                                                              np.around(
                                                                                                  (dt % 3600) % 60, 3)))

        return cropped_timeseries

    @staticmethod
    def _crop_timeseries(it, timeseries, example_data, delta_ts, timeseries_delta_t):
        """
        Removes timestamps from a timeseries that are within a data gap.
        """
        cropped_timeseries = []
        prev_i = 0
        for timestamp in timeseries:
            found = False
            for i in range(prev_i, len(example_data) - 1):
                dp_i = example_data[i]
                dp_i1 = example_data[i + 1]
                if dp_i.timestamp <= timestamp < dp_i1.timestamp:
                    found = True
                    prev_i = i
                    # Add timestamp, if not in gap
                    delta_t = delta_ts[i]
                    if delta_t <= timeseries_delta_t:
                        cropped_timeseries.append(timestamp)
                        break
            if not found:
                for i in range(len(example_data) - 1):
                    dp_i = example_data[i]
                    dp_i1 = example_data[i + 1]
                    if dp_i.timestamp <= timestamp < dp_i1.timestamp:
                        prev_i = i
                        # Add timestamp, if not in gap
                        delta_t = delta_ts[i]
                        if delta_t <= timeseries_delta_t:
                            cropped_timeseries.append(timestamp)
                            break

        return it, cropped_timeseries

    @staticmethod
    def _datetime_to_timestamp(dt):
        return calendar.timegm(dt.timetuple())

    @staticmethod
    def get_value_ids_for_histogram_type(histogram_type):
        """
        Returns the value ids for the given histogram type.
        :param histogram_type: HistogramType
        :return: Tuple, (value_id_x, value_id_y)
        """
        if histogram_type is HistogramType.CELL_C_RATE:
            return 1288, 0
        elif histogram_type is HistogramType.PACK_TEMP_MAX:
            return 1209, 0
        elif histogram_type is HistogramType.PACK_TEMP_MIN:
            return 1208, 0
        elif histogram_type is HistogramType.PACK_TEMP_DELTA:
            return 1289, 0
        elif histogram_type is HistogramType.PACK_SOC:
            return 900, 0
        elif histogram_type is HistogramType.PACK_DOD:
            return 1290, 0
        elif histogram_type is HistogramType.VEHICLE_SPEED:
            return 4, 0
        elif histogram_type is HistogramType.TRACK_DURATION:
            return 1291, 0
        elif histogram_type is HistogramType.IDLE_PERIOD_DURATION:
            return 1292, 0
        elif histogram_type is HistogramType.PACK_VOLTAGE:
            return 1200, 0
        elif histogram_type is HistogramType.CELL_VOLTAGE_MAX:
            return 1293, 0
        elif histogram_type is HistogramType.CELL_VOLTAGE_MIN:
            return 1294, 0
        elif histogram_type is HistogramType.CELL_VOLTAGE_DELTA:
            return 1295, 0
        elif histogram_type is HistogramType.CHG_POWER:
            return 1297, 0
        elif histogram_type is HistogramType.CHG_DURATION:
            return 1296, 0
        elif histogram_type is HistogramType.CHG_EOC:
            return 1298, 0
        elif histogram_type is HistogramType.AMBIENT_AIR_TEMP:
            return 15, 0
        elif histogram_type is HistogramType.COOLANT_TEMP_INVERTER_INLET:
            return 1269, 0
        elif histogram_type is HistogramType.PACK_COOLANT_TEMP_INLET:
            return 1272, 0
        elif histogram_type is HistogramType.PACK_COOLANT_TEMP_OUTLET:
            return 1273, 0
        elif histogram_type is HistogramType.HEATPUMP_POWER:
            return 42, 0
        elif histogram_type is HistogramType.ROTOR_REAR_TEMP:
            return 962, 0
        elif histogram_type is HistogramType.STATOR_REAR_TEMP:
            return 961, 0
        elif histogram_type is HistogramType.INTERIOR_TEMP:
            return 43, 0
        elif histogram_type is HistogramType.PTC1_CURRENT:
            return 1205, 0
        elif histogram_type is HistogramType.PTC2_CURRENT:
            return 1206, 0
        elif histogram_type is HistogramType.PTC_VOLTAGE:
            return 1207, 0
        elif histogram_type is HistogramType.REAR_INVERTER_TEMP:
            return 963, 0
        elif histogram_type is HistogramType.TRAVELED_DISTANCE:
            return 1299, 0
        elif histogram_type is HistogramType.AUXILIARIES_POWER:
            return 56, 0
        elif histogram_type is HistogramType.PTC1_POWER:
            return 1300, 0
        elif histogram_type is HistogramType.PTC2_POWER:
            return 1301, 0
        elif histogram_type is HistogramType.C_RATE_PEAK_ANALYSIS_AMPL:
            return 1302, 0
        elif histogram_type is HistogramType.C_RATE_PEAK_ANALYSIS_FREQ:
            return 1303, 0
        elif histogram_type is HistogramType.HIST2D_C_RATE__PACK_TEMP_MAX:
            return BEVUsage.get_value_ids_for_histogram_type(HistogramType.CELL_C_RATE)[0], \
                BEVUsage.get_value_ids_for_histogram_type(HistogramType.PACK_TEMP_MAX)[0]
        elif histogram_type is HistogramType.HIST2D_C_RATE__PACK_TEMP_DELTA:
            return BEVUsage.get_value_ids_for_histogram_type(HistogramType.CELL_C_RATE)[0], \
                BEVUsage.get_value_ids_for_histogram_type(HistogramType.PACK_TEMP_DELTA)[0]
        elif histogram_type is HistogramType.HIST2D_C_RATE__PACK_SOC:
            return BEVUsage.get_value_ids_for_histogram_type(HistogramType.CELL_C_RATE)[0], \
                BEVUsage.get_value_ids_for_histogram_type(HistogramType.PACK_SOC)[0]
        elif histogram_type is HistogramType.HIST2D_C_RATE__VEHICLE_SPEED:
            return BEVUsage.get_value_ids_for_histogram_type(HistogramType.CELL_C_RATE)[0], \
                BEVUsage.get_value_ids_for_histogram_type(HistogramType.VEHICLE_SPEED)[0]
        elif histogram_type is HistogramType.HIST2D_PACK_TEMP_MAX__PACK_SOC:
            return BEVUsage.get_value_ids_for_histogram_type(HistogramType.PACK_TEMP_MAX)[0], \
                BEVUsage.get_value_ids_for_histogram_type(HistogramType.PACK_SOC)[0]
        elif histogram_type is HistogramType.CHG_HIST2D_POWER__PACK_TEMP_MAX:
            return BEVUsage.get_value_ids_for_histogram_type(HistogramType.CHG_POWER)[0], \
                BEVUsage.get_value_ids_for_histogram_type(HistogramType.PACK_TEMP_MAX)[0]
        elif histogram_type is HistogramType.CHG_HIST2D_POWER__PACK_TEMP_DELTA:
            return BEVUsage.get_value_ids_for_histogram_type(HistogramType.CHG_POWER)[0], \
                BEVUsage.get_value_ids_for_histogram_type(HistogramType.PACK_TEMP_DELTA)[0]
        elif histogram_type is HistogramType.CHG_HIST2D_POWER__SOC:
            return BEVUsage.get_value_ids_for_histogram_type(HistogramType.CHG_POWER)[0], \
                BEVUsage.get_value_ids_for_histogram_type(HistogramType.PACK_SOC)[0]
        else:
            raise NotImplementedError("Missing value id data for histogram type '" + str(histogram_type.name) + "'!")

    @staticmethod
    def _get_data_gaps(data, min_gap_len=0):
        # Determine the measurement frequency
        datetime_delta_ts = [data[i + 1].timestamp - data[i].timestamp
                             for i in range(len(data) - 1)]
        delta_ts = [datetime_delta_t.seconds + datetime_delta_t.microseconds / 1000000
                    for datetime_delta_t in datetime_delta_ts]
        measurement_frequency = BEVUsage._round_to_significant_digits(float(np.median(delta_ts)), 1)

        gaps = []
        gap_start = None
        for i in range(len(data) - 1):
            # Filter out gaps in the data
            delta_t = delta_ts[i]
            if delta_t > measurement_frequency:
                if gap_start is None:
                    gap_start = data[i].timestamp
            else:
                if gap_start is not None:
                    if ((data[i].timestamp - gap_start) / timedelta(microseconds=1)) / 10e6 > min_gap_len:
                        gaps.append((gap_start, data[i].timestamp))
                    gap_start = None

        return gaps
