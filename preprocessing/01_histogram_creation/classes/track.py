# Class Track
# Contains all data of a vehicle test drive
# Author: Lukas L. Köning

import logging
import os
from datetime import datetime, timedelta
from typing import List

import numpy as np
import psycopg2
import scipy.signal as scisig

from .bev_usage import BEVUsage, HistogramType
from .datapoint import Datapoint, RoadType
from .exceptions import *
from .histogram import Histogram, Histogram2D, Bins


class Track(BEVUsage):
    # Hyperparameter
    IMPLEMENTED_HISTOGRAMS = [
        HistogramType.CELL_C_RATE,
        HistogramType.PACK_TEMP_MAX,
        HistogramType.PACK_TEMP_MIN,
        HistogramType.PACK_TEMP_DELTA,
        HistogramType.PACK_SOC,
        HistogramType.PACK_DOD,
        HistogramType.VEHICLE_SPEED,
        HistogramType.TRACK_DURATION,
        HistogramType.IDLE_PERIOD_DURATION,
        HistogramType.PACK_VOLTAGE,
        HistogramType.CELL_VOLTAGE_MAX,
        HistogramType.CELL_VOLTAGE_MIN,
        HistogramType.CELL_VOLTAGE_DELTA,
        HistogramType.HIST2D_C_RATE__PACK_TEMP_MAX,
        HistogramType.HIST2D_C_RATE__PACK_TEMP_DELTA,
        HistogramType.HIST2D_C_RATE__PACK_SOC,
        HistogramType.HIST2D_C_RATE__VEHICLE_SPEED,
        HistogramType.HIST2D_PACK_TEMP_MAX__PACK_SOC,
        HistogramType.AMBIENT_AIR_TEMP,
        HistogramType.COOLANT_TEMP_INVERTER_INLET,
        HistogramType.PACK_COOLANT_TEMP_INLET,
        HistogramType.PACK_COOLANT_TEMP_OUTLET,
        HistogramType.ROTOR_REAR_TEMP,
        HistogramType.STATOR_REAR_TEMP,
        HistogramType.INTERIOR_TEMP,
        HistogramType.PTC1_CURRENT,
        HistogramType.PTC2_CURRENT,
        HistogramType.PTC_VOLTAGE,
        HistogramType.REAR_INVERTER_TEMP,
        HistogramType.TRAVELED_DISTANCE,
        HistogramType.AUXILIARIES_POWER,
        HistogramType.PTC1_POWER,
        HistogramType.PTC2_POWER,
        HistogramType.C_RATE_PEAK_ANALYSIS_AMPL,
        HistogramType.C_RATE_PEAK_ANALYSIS_FREQ
    ]

    def __init__(self, track_id, start, end, vehicle, processed, prior_usage_end=None, distance=0):
        super().__init__(database_id=track_id, start=start, end=end, vehicle=vehicle, processed=processed,
                         prior_usage_end=prior_usage_end)
        self.traveled_distance = distance

    def analyze(self, desired_histograms: List[HistogramType], plot=False):
        """
        This function analyses the track and creates histograms.
        :param desired_histograms: List of HistogramTypes of which histograms should be calculated
        :param plot: Bool, plot the calculated histograms?
        :return:
        """
        logging.info("Analyzing track '" + str(self.id) + "'...")

        # Check, if histograms have already been calculated
        remaining_histogram_types = self._check_if_already_processed(desired_histograms)
        if len(remaining_histogram_types) == len(desired_histograms):
            logging.info("--> Calculating all desired histograms, since none has been already calculated.")
        elif len(remaining_histogram_types) == 0:
            logging.info("--> All requested histograms for this track has already been calculated!")
            return
        else:
            logging.info("--> Calculating " + str(len(remaining_histogram_types)) + " of " +
                         str(len(desired_histograms)) + " requested histogram, since some already have been "
                                                        "calculated.")
            desired_histograms = remaining_histogram_types

        # Make sure, all data is available
        if len(self.value_id_dict.keys()) == 0:
            self.get_value_id_data()

        if len(self.can_data.keys()) == 0:
            self.fetch_can_data()

        # Check, if track is complete (vehicle speed is needed for track analysis)
        if not self.is_complete() or "vehicle_speed" not in self.can_data.keys():
            raise RawDataError("Track is not complete and therefore can not be processed.")

        # Remove charging processes from track
        self.remove_charging_from_data()
        logging.info("--> Removed charging artifacts successfully.")

        # Calculate Histograms
        for hist_type in desired_histograms:
            try:
                self.calculate_histogram(histogram_type=hist_type, plot=plot, keep_cache=True)
            except RawDataError as rde:
                logging.warning(
                    "--> Histogram of type '" + str(hist_type.name) + "' could not be created. Reason: " + str(rde))
            except HistogramError as he:
                logging.error("--> Histogram of type '" + str(
                    hist_type.name) + "' could not be created. Reason: Histogram calculation failed with error '" + str(
                    he) + "'")

        # Empty cache after successful calculation of all histograms
        self.cache = {}

    def save_histograms_to_database(self, force_overwrite=False):
        """
        Saves the calculated Histograms from the internal list self.histograms_list to the database
        :return:
        """
        conn = psycopg2.connect(database="mobtrack",
                                user=os.environ["db_user"],
                                password=os.environ["db_pwd"],
                                host=os.environ["host"],
                                port="5432")
        conn.autocommit = True
        cur = conn.cursor()

        # Get all already processed tracks
        sql = '''SELECT value_id_x, value_id_y, road_type FROM track.track_ev_histograms WHERE track_id=%s'''
        sql_parameters = [self.id]
        cur.execute(sql, sql_parameters)
        existing_histograms = cur.fetchall()

        # Get existing bin combinations
        cur.execute('''SELECT id, maximum, minimum, stepsize FROM track.histogram_bins''')
        bins_result = cur.fetchall()
        existing_bins_dict = {}
        for b in bins_result:
            existing_bins_dict[b[0]] = Bins(maximum=b[1], minimum=b[2], step_size=b[3])

        for hist in self.histogram_list:
            if type(hist) == Histogram:
                if (hist.data_type.id, 0, hist.road_type.name.lower()) in existing_histograms and not force_overwrite:
                    continue
                elif (hist.data_type.id, 0, hist.road_type.name.lower()) in existing_histograms and force_overwrite:
                    if hist.bins in existing_bins_dict.values():
                        bins_id = self._get_key_for_value(existing_bins_dict, hist.bins)
                    else:
                        insert_bin_sql = '''INSERT INTO track.histogram_bins (maximum, minimum, stepsize) values (%s, %s, %s) ON CONFLICT ON CONSTRAINT histogram_bins_un DO UPDATE SET maximum = excluded.maximum RETURNING id;'''
                        insert_bin_sql_parameters = [hist.bins.maximum, hist.bins.minimum, hist.bins.step_size]
                        cur.execute(insert_bin_sql, insert_bin_sql_parameters)
                        bins_id = cur.fetchone()[0]

                    sql_query = '''UPDATE track.track_ev_histograms 
                    SET counts = %s, bins_id_x = %s, bins_id_y = 0, time_computed = now(), road_type = %s
                    WHERE track_id=%s AND value_id_x=%s AND value_id_y=0'''
                    sql_parameters = [hist.counts.tolist(), bins_id,
                                      hist.road_type.name.lower(), self.id, hist.data_type.id]
                else:
                    if hist.bins in existing_bins_dict.values():
                        bins_id = self._get_key_for_value(existing_bins_dict, hist.bins)
                    else:
                        insert_bin_sql = '''INSERT INTO track.histogram_bins (maximum, minimum, stepsize) values (%s, %s, %s) ON CONFLICT ON CONSTRAINT histogram_bins_un DO UPDATE SET maximum = excluded.maximum RETURNING id;'''
                        insert_bin_sql_parameters = [hist.bins.maximum, hist.bins.minimum, hist.bins.step_size]
                        cur.execute(insert_bin_sql, insert_bin_sql_parameters)
                        bins_id = cur.fetchone()[0]

                    sql_query = '''INSERT INTO track.track_ev_histograms 
                    (track_id, value_id_x, value_id_y, counts, bins_id_x, bins_id_y, time_computed, road_type) 
                    VALUES (%s, %s, 0, %s, %s, 0, now(), %s)'''
                    sql_parameters = [self.id, hist.data_type.id, hist.counts.tolist(),
                                      bins_id, hist.road_type.name.lower()]
                cur.execute(sql_query, sql_parameters)
                conn.commit()
            elif type(hist) == Histogram2D:
                if (hist.x_data_type.id, hist.y_data_type.id, hist.road_type.name.lower()) in existing_histograms \
                        and not force_overwrite:
                    continue
                elif (hist.x_data_type.id, hist.y_data_type.id, hist.road_type.name.lower()) in existing_histograms \
                        and force_overwrite:
                    # Check if x bins already exist
                    if hist.x_bins in existing_bins_dict.values():
                        bins_id_x = self._get_key_for_value(existing_bins_dict, hist.x_bins)
                    else:
                        insert_bin_sql = '''INSERT INTO track.histogram_bins (maximum, minimum, stepsize) values (%s, %s, %s) ON CONFLICT ON CONSTRAINT histogram_bins_un DO UPDATE SET maximum = excluded.maximum RETURNING id;'''
                        insert_bin_sql_parameters = [hist.x_bins.maximum, hist.x_bins.minimum, hist.x_bins.step_size]
                        cur.execute(insert_bin_sql, insert_bin_sql_parameters)
                        bins_id_x = cur.fetchone()[0]

                    # Check if y bins already exist
                    if hist.y_bins in existing_bins_dict.values():
                        bins_id_y = self._get_key_for_value(existing_bins_dict, hist.y_bins)
                    else:
                        insert_bin_sql = '''INSERT INTO track.histogram_bins (maximum, minimum, stepsize) values (%s, %s, %s) ON CONFLICT ON CONSTRAINT histogram_bins_un DO UPDATE SET maximum = excluded.maximum RETURNING id;'''
                        insert_bin_sql_parameters = [hist.y_bins.maximum, hist.y_bins.minimum,
                                                     hist.y_bins.step_size]
                        cur.execute(insert_bin_sql, insert_bin_sql_parameters)
                        bins_id_y = cur.fetchone()[0]

                    sql_query = '''UPDATE track.track_ev_histograms SET 
                    counts = %s, bins_id_x = %s, bins_id_y = %s, time_computed = now(), road_type = %s
                    WHERE track_id=%s AND value_id_x=%s AND value_id_y=%s'''
                    sql_parameters = [hist.counts.tolist(), bins_id_x, bins_id_y,
                                      hist.road_type.name.lower(), self.id, hist.x_data_type.id, hist.y_data_type.id]
                else:
                    # Check if x bins already exist
                    if hist.x_bins in existing_bins_dict.values():
                        bins_id_x = self._get_key_for_value(existing_bins_dict, hist.x_bins)
                    else:
                        insert_bin_sql = '''INSERT INTO track.histogram_bins (maximum, minimum, stepsize) values (%s, %s, %s) ON CONFLICT ON CONSTRAINT histogram_bins_un DO UPDATE SET maximum = excluded.maximum RETURNING id;'''
                        insert_bin_sql_parameters = [hist.x_bins.maximum, hist.x_bins.minimum, hist.x_bins.step_size]
                        cur.execute(insert_bin_sql, insert_bin_sql_parameters)
                        bins_id_x = cur.fetchone()[0]

                    # Check if y bins already exist
                    if hist.y_bins in existing_bins_dict.values():
                        bins_id_y = self._get_key_for_value(existing_bins_dict, hist.y_bins)
                    else:
                        insert_bin_sql = '''INSERT INTO track.histogram_bins (maximum, minimum, stepsize) values (%s, %s, %s) ON CONFLICT ON CONSTRAINT histogram_bins_un DO UPDATE SET maximum = excluded.maximum RETURNING id;'''
                        insert_bin_sql_parameters = [hist.y_bins.maximum, hist.y_bins.minimum,
                                                     hist.y_bins.step_size]
                        cur.execute(insert_bin_sql, insert_bin_sql_parameters)
                        bins_id_y = cur.fetchone()[0]

                    sql_query = '''INSERT INTO track.track_ev_histograms 
                    (track_id, value_id_x, value_id_y, counts, bins_id_x, bins_id_y, time_computed, road_type) 
                    VALUES (%s, %s, %s, %s, %s, %s, now(), %s)'''
                    sql_parameters = [self.id, hist.x_data_type.id, hist.y_data_type.id,
                                      hist.counts.tolist(), bins_id_x, bins_id_y,
                                      hist.road_type.name.lower()]
                cur.execute(sql_query, sql_parameters)
                conn.commit()
        cur.close()
        conn.close()

    # noinspection PyTypeChecker
    def calculate_histogram(self, histogram_type: HistogramType, plot=False, keep_cache=False):
        """
        Calculates the histogram for the given histogram type and add it to the internal list
        :param histogram_type: HistogramType, type of the histogram that should be created
        :param plot: Bool, plot the histogram?
        :param keep_cache: Bool, keep the track cache or delete it?
        :return: nothing
        """
        # Set up some variables
        temperature_types = [HistogramType.PACK_TEMP_MAX, HistogramType.PACK_TEMP_MIN, HistogramType.PACK_TEMP_DELTA]
        cell_voltage_types = [HistogramType.CELL_VOLTAGE_MAX, HistogramType.CELL_VOLTAGE_MIN,
                              HistogramType.CELL_VOLTAGE_DELTA]
        title_prefix = "[Fahren] "

        # C-Rate
        if histogram_type is HistogramType.CELL_C_RATE:
            if "hv_battery_current" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            c_rates = [Datapoint("cell_c_rate", float(i[0]) / self.vehicle.hv_cap, i[1]) for i in
                       [(dp.value, dp.timestamp) for dp in self.can_data["hv_battery_current"]]]
            c_rate_hist = self._create_custom_histogram(self.value_id_dict[1288],
                                                        c_rates, self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])
            if plot:
                c_rate_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(c_rate_hist)
            logging.info("--> Calculated histogram of c-rate successfully.")

        elif histogram_type is HistogramType.C_RATE_PEAK_ANALYSIS_AMPL:
            if "hv_battery_current" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            c_rates = [Datapoint("cell_c_rate", float(i[0]) / self.vehicle.hv_cap, i[1]) for i in
                       [(dp.value, dp.timestamp) for dp in self.can_data["hv_battery_current"]]]
            peaks = self.analyze_peaks(data=c_rates,
                                       split_lengths=[100, 250, 500, 1000],
                                       min_prominences=[0, 0.005, 0.013, 0.15],
                                       overlap=2)
            amplitudes = [Datapoint("cell_c_rate_peak_ampl", peak["amplitude"], c_rates[key].timestamp) for key, peak in
                          peaks.items()]

            c_rate_amplitude_hist = self._create_custom_histogram(self.value_id_dict[1302],
                                                                  amplitudes,
                                                                  self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])
            if plot:
                c_rate_amplitude_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(c_rate_amplitude_hist)
            logging.info("--> Calculated histogram of c-rate peak amplitudes successfully.")

        elif histogram_type is HistogramType.C_RATE_PEAK_ANALYSIS_FREQ:
            if "hv_battery_current" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            c_rates = [Datapoint("cell_c_rate", float(i[0]) / self.vehicle.hv_cap, i[1]) for i in
                       [(dp.value, dp.timestamp) for dp in self.can_data["hv_battery_current"]]]
            peaks = self.analyze_peaks(data=c_rates,
                                       split_lengths=[100, 250, 500, 1000],
                                       min_prominences=[0, 0.005, 0.013, 0.15],
                                       overlap=2)
            frequencies = [Datapoint("cell_c_rate_peak_freq", peak["frequency"], c_rates[key].timestamp) for key, peak
                           in peaks.items()]

            c_rate_frequency_hist = self._create_custom_histogram(self.value_id_dict[1303],
                                                                  frequencies,
                                                                  self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])
            if plot:
                c_rate_frequency_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(c_rate_frequency_hist)
            logging.info("--> Calculated histogram of c-rate peak frequency successfully.")

        # Temperatures
        elif histogram_type in temperature_types:
            max_temp = min_temp = None

            if "max_temp" in self.cache.keys() and "min_temp" in self.cache.keys():
                max_temp = self.cache["max_temp"]
                min_temp = self.cache["min_temp"]
            elif len([s for s in self.can_data.keys() if "pack_temp_" in s and s != "pack_temp_0"]) > 1:
                # Try to calculate max and min temperature from pack temperatures
                # Collect pack temperatures data
                pack_temp_data = []
                for pack_temp_string in [s for s in self.can_data.keys() if
                                         "pack_temp_" in s and s != "pack_temp_0"]:
                    pack_temp_data.append(
                        sorted(self.can_data[pack_temp_string], key=lambda x: x.timestamp, reverse=False))

                # Prepare timeseries to make comparison between pack temperatures possible
                latest_first_dp_timestamp = max(
                    [pack_temp_data_i[0].timestamp for pack_temp_data_i in pack_temp_data])
                earliest_last_dp_timestamp = min(
                    [pack_temp_data_i[-1].timestamp for pack_temp_data_i in pack_temp_data])
                if earliest_last_dp_timestamp < latest_first_dp_timestamp:
                    raise RawDataError("Not enough data for calculations!")
                timeseries = np.arange(start=latest_first_dp_timestamp, stop=earliest_last_dp_timestamp,
                                       step=timedelta(seconds=1)).astype(datetime)
                timeseries = Track._add_data_gaps_to_timeseries(self.can_data["hv_battery_current"], timeseries)

                # Match pack temperatures data to timeseries
                pack_temp_data_in_timeseries = []
                for i in range(len(pack_temp_data)):
                    pack_temp_data_in_timeseries.append(
                        Track._match_data_to_timeseries(pack_temp_data[i], timeseries))

                # Calculate maximal and minimal temperature for each timestep
                max_temp = []
                min_temp = []
                for i in range(len(timeseries) - 1):
                    max_temp_dp = Datapoint(variable_name="hv_temp_max",
                                            variable_value=max([pack_temp_j[i].value
                                                                for pack_temp_j in pack_temp_data_in_timeseries]),
                                            time_of_measurement=timeseries[i],
                                            road_type=pack_temp_data_in_timeseries[0][i].road_type)
                    min_temp_dp = Datapoint(variable_name="hv_temp_min",
                                            variable_value=min([pack_temp_j[i].value
                                                                for pack_temp_j in pack_temp_data_in_timeseries]),
                                            time_of_measurement=timeseries[i],
                                            road_type=pack_temp_data_in_timeseries[0][i].road_type)
                    max_temp.append(max_temp_dp)
                    min_temp.append(min_temp_dp)
            elif "hv_temp_max" in self.can_data.keys() \
                    and "hv_temp_min" in self.can_data.keys() \
                    and 0 not in [dp.value for dp in self.can_data["hv_temp_min"]] \
                    and 0 not in [dp.value for dp in self.can_data[
                "hv_temp_max"]]:  # Prevent calculations with data from not working sensors
                raw_max_temp = self.can_data["hv_temp_max"]
                raw_min_temp = self.can_data["hv_temp_min"]
                raw_temp_data = [raw_max_temp, raw_min_temp]

                # Match temperatures to timeseries to calculate later the correct timedelta
                latest_first_dp_timestamp = max(
                    [pack_temp_data_i[0].timestamp for pack_temp_data_i in raw_temp_data])
                earliest_last_dp_timestamp = min(
                    [pack_temp_data_i[-1].timestamp for pack_temp_data_i in raw_temp_data])
                if earliest_last_dp_timestamp < latest_first_dp_timestamp:
                    raise RawDataError("Not enough data for calculations!")
                timeseries = np.arange(start=latest_first_dp_timestamp, stop=earliest_last_dp_timestamp,
                                       step=timedelta(seconds=1)).astype(datetime)
                timeseries = Track._add_data_gaps_to_timeseries(self.can_data["hv_battery_current"], timeseries)

                max_temp = Track._match_data_to_timeseries(raw_max_temp, timeseries)
                min_temp = Track._match_data_to_timeseries(raw_min_temp, timeseries)
            elif histogram_type is HistogramType.PACK_TEMP_MAX and "hv_temp_max" in self.can_data.keys() and 0 not in [
                dp.value for dp in self.can_data["hv_temp_max"]]:
                max_temp = self.can_data["hv_temp_max"]
            elif histogram_type is HistogramType.PACK_TEMP_MIN and "hv_temp_min" in self.can_data.keys() and 0 not in [
                dp.value for dp in self.can_data["hv_temp_min"]]:
                min_temp = self.can_data["hv_temp_min"]
            else:
                raise RawDataError("Missing data for calculations!")

            # Save data into cache for later calculations
            if max_temp is not None:
                self.cache["max_temp"] = max_temp
            if min_temp is not None:
                self.cache["min_temp"] = min_temp

            # Calculate histogram
            if histogram_type is HistogramType.PACK_TEMP_MAX:
                max_temp_hist = self._create_custom_histogram(data_type=self.value_id_dict[1209],
                                                              data=max_temp,
                                                              bins=self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])
                if plot:
                    max_temp_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
                self.histogram_list.append(max_temp_hist)
                logging.info("--> Calculated histogram of maximal temperature successfully.")
            elif histogram_type is HistogramType.PACK_TEMP_MIN:
                min_temp_hist = self._create_custom_histogram(data_type=self.value_id_dict[1208],
                                                              data=min_temp,
                                                              bins=self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])
                if plot:
                    min_temp_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
                self.histogram_list.append(min_temp_hist)
                logging.info("--> Calculated histogram of minimal temperature successfully.")
            elif histogram_type is HistogramType.PACK_TEMP_DELTA:
                delta_temp_hist = self._create_custom_histogram(data_type=self.value_id_dict[1289],
                                                                data=list(np.subtract(max_temp, min_temp)),
                                                                bins=self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])
                if plot:
                    delta_temp_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
                self.histogram_list.append(delta_temp_hist)
                logging.info("--> Calculated histogram of temperature difference successfully.")

        # SOC
        elif histogram_type is HistogramType.PACK_SOC:
            if "hv_soc" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            soc_hist = self._create_histogram(data_type_name="hv_soc",
                                              bins=self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])
            if plot:
                soc_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(soc_hist)
            logging.info("--> Calculated histogram of SOC successfully.")

        # DOD
        elif histogram_type is HistogramType.PACK_DOD:
            if "hv_soc" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            dod = self.can_data["hv_soc"][0] - self.can_data["hv_soc"][-1]

            # Check for valid dod value
            if dod.value < 0 and abs(dod.value) < 0.5:
                dod.value = 0
            elif dod.value < 0 and abs(dod.value) > 0.5:
                raise RawDataError("Negative value for DOD ({}%). Please check track!".format(dod.value))

            dod_hist = self._create_custom_histogram(data_type=self.value_id_dict[1290],
                                                     data=[dod],
                                                     bins=self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])
            if plot:
                dod_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(dod_hist)
            logging.info("--> Calculated histogram of DOD successfully.")

        # Speed
        elif histogram_type is HistogramType.VEHICLE_SPEED:
            if "vehicle_speed" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            # Filter out sensor errors
            vehicle_speed = [dp for dp in self.can_data["vehicle_speed"] if 0 < dp.value <= 254]

            speed_hist = self._create_custom_histogram(data_type=self.value_id_dict[4],
                                                       data=vehicle_speed,
                                                       bins=self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])

            if plot:
                speed_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(speed_hist)
            logging.info("--> Calculated histogram of vehicle speed successfully.")

        # Track duration
        elif histogram_type is HistogramType.TRACK_DURATION:
            duration = (datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S") -
                        datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")).total_seconds()
            duration_hist = self._create_custom_histogram(data_type=self.value_id_dict[1291],
                                                          data=[Datapoint(variable_name="track_duration",
                                                                          variable_value=duration / 60,
                                                                          time_of_measurement=datetime.strptime(
                                                                              self.end_time, "%Y-%m-%d %H:%M:%S"),
                                                                          road_type=RoadType.UNKNOWN)],
                                                          bins=self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])
            if plot:
                duration_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(duration_hist)
            logging.info("--> Calculated histogram of track duration successfully.")

        # idle period duration
        elif histogram_type is HistogramType.IDLE_PERIOD_DURATION:
            if self.prior_usage_end is None:
                raise RawDataError("Missing data (end time of prior usage)!")

            duration = (datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S") -
                        datetime.strptime(self.prior_usage_end, "%Y-%m-%d %H:%M:%S")).total_seconds()
            idle_hist = self._create_custom_histogram(
                data_type=self.value_id_dict[1292],
                data=[Datapoint(variable_name="idle_period_duration",
                                variable_value=duration / 60,
                                time_of_measurement=datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S"),
                                road_type=RoadType.UNKNOWN)],
                bins=self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])
            if plot:
                idle_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(idle_hist)
            logging.info("--> Calculated histogram of idle period duration successfully.")

        # battery voltage
        elif histogram_type is HistogramType.PACK_VOLTAGE:
            if "hv_battery_voltage" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            # Filter out sensor errors
            hv_battery_voltage = [dp for dp in self.can_data["hv_battery_voltage"] if 0 < dp.value <= 900]

            hv_voltage_hist = self._create_custom_histogram(data_type=self.value_id_dict[1200],
                                                            data=hv_battery_voltage,
                                                            bins=self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])

            if plot:
                hv_voltage_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(hv_voltage_hist)
            logging.info("--> Calculated histogram of battery pack voltages successfully.")

        # cell voltages
        elif histogram_type in cell_voltage_types:
            if "max_cell_voltage" in self.cache.keys() and "min_cell_voltage" in self.cache.keys():
                max_cell_voltage = self.cache["max_cell_voltage"]
                min_cell_voltage = self.cache["min_cell_voltage"]
            elif len([s for s in self.can_data.keys() if "cell_voltage_" in s]) > 1:
                # Collect cell voltage data
                cell_voltage_data = []
                for cell_voltage_string in [s for s in self.can_data.keys() if "cell_voltage_" in s]:
                    # Remove sensor errors
                    filtered_data = [dp for dp in self.can_data[cell_voltage_string] if 0 < dp.value < 6]
                    if len(filtered_data) > 5:
                        cell_voltage_data.append(sorted(filtered_data, key=lambda x: x.timestamp, reverse=False))

                if len(cell_voltage_data) < 2:
                    raise RawDataError("Not enough data for calculations!")

                # Prepare timeseries to make comparison between cell voltages possible
                latest_first_dp_timestamp = max(
                    [pack_temp_data_i[0].timestamp for pack_temp_data_i in cell_voltage_data])
                earliest_last_dp_timestamp = min(
                    [pack_temp_data_i[-1].timestamp for pack_temp_data_i in cell_voltage_data])
                if earliest_last_dp_timestamp < latest_first_dp_timestamp:
                    raise RawDataError("Not enough data for calculations!")
                timeseries = np.arange(start=latest_first_dp_timestamp, stop=earliest_last_dp_timestamp,
                                       step=timedelta(seconds=1)).astype(datetime)
                timeseries = Track._add_data_gaps_to_timeseries(self.can_data["hv_battery_current"], timeseries)

                # Match cell voltage data to timeseries
                cell_voltage_data_in_timeseries = []
                for i in range(len(cell_voltage_data)):
                    cell_voltage_data_in_timeseries.append(
                        Track._match_data_to_timeseries(cell_voltage_data[i], timeseries))

                # Calculate maximal and minimal cell voltage per timestamp
                max_cell_voltage = []
                min_cell_voltage = []
                for i in range(len(timeseries) - 1):
                    max_cell_voltage_dp = Datapoint(variable_name="cell_voltage_max",
                                                    variable_value=max(
                                                        [cell_voltage_j[i].value for cell_voltage_j in
                                                         cell_voltage_data_in_timeseries]),
                                                    time_of_measurement=timeseries[i],
                                                    road_type=cell_voltage_data_in_timeseries[0][i].road_type)
                    min_cell_voltage_dp = Datapoint(variable_name="cell_voltage_min",
                                                    variable_value=min(
                                                        [cell_voltage_j[i].value for cell_voltage_j in
                                                         cell_voltage_data_in_timeseries]),
                                                    time_of_measurement=timeseries[i],
                                                    road_type=cell_voltage_data_in_timeseries[0][i].road_type)
                    max_cell_voltage.append(max_cell_voltage_dp)
                    min_cell_voltage.append(min_cell_voltage_dp)
            else:
                raise RawDataError("Missing data for calculations!")

            # Save data to cache for future calculations
            self.cache["max_cell_voltage"] = max_cell_voltage
            self.cache["min_cell_voltage"] = min_cell_voltage

            # Calculate histogram
            if histogram_type is HistogramType.CELL_VOLTAGE_MAX:
                max_cell_voltage_hist = self._create_custom_histogram(data_type=self.value_id_dict[1293],
                                                                      data=max_cell_voltage,
                                                                      bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                          histogram_type][0])
                if plot:
                    max_cell_voltage_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
                self.histogram_list.append(max_cell_voltage_hist)
                logging.info("--> Calculated histogram of maximal cell voltage successfully.")
            elif histogram_type is HistogramType.CELL_VOLTAGE_MIN:
                min_cell_voltage_hist = self._create_custom_histogram(data_type=self.value_id_dict[1294],
                                                                      data=min_cell_voltage,
                                                                      bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                          histogram_type][0])
                if plot:
                    min_cell_voltage_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
                self.histogram_list.append(min_cell_voltage_hist)
                logging.info("--> Calculated histogram of minimal cell voltage successfully.")
            elif histogram_type is HistogramType.CELL_VOLTAGE_DELTA:
                delta_cell_voltage_hist = self._create_custom_histogram(data_type=self.value_id_dict[1295],
                                                                        data=list(np.subtract(max_cell_voltage,
                                                                                              min_cell_voltage)),
                                                                        bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                            histogram_type][0])
                if plot:
                    delta_cell_voltage_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
                self.histogram_list.append(delta_cell_voltage_hist)
                logging.info("--> Calculated histogram of cell voltage differences successfully.")

        # ambient air temperature
        elif histogram_type is HistogramType.AMBIENT_AIR_TEMP:
            if "ambient_air_temp" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            ambient_air_hist = self._create_histogram(data_type_name="ambient_air_temp",
                                                      bins=self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])
            if plot:
                ambient_air_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(ambient_air_hist)
            logging.info("--> Calculated histogram of ambient air temperature successfully.")

        # coolant temperature at inverter inlet
        elif histogram_type is HistogramType.COOLANT_TEMP_INVERTER_INLET:
            if "coolant_temp_inverter_inlet" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            coolant_temp_inverter_inlet_hist = self._create_histogram(data_type_name="coolant_temp_inverter_inlet",
                                                                      bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                          histogram_type][0])
            if plot:
                coolant_temp_inverter_inlet_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(coolant_temp_inverter_inlet_hist)
            logging.info("--> Calculated histogram of coolant temperature at inverter inlet successfully.")

        # coolant temperature at pack inlet
        elif histogram_type is HistogramType.PACK_COOLANT_TEMP_INLET:
            if "hv_battery_temp_inlet" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            # Filter out sensor errors
            hv_battery_temp_inlet = [dp for dp in self.can_data["hv_battery_temp_inlet"] if -40 < dp.value <= 100]

            hv_battery_temp_inlet_hist = self._create_custom_histogram(data_type=self.value_id_dict[1272],
                                                                       data=hv_battery_temp_inlet,
                                                                       bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                           histogram_type][0])
            if plot:
                hv_battery_temp_inlet_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(hv_battery_temp_inlet_hist)
            logging.info("--> Calculated histogram of coolant temperature at battery pack inlet successfully.")

        # coolant temperature at pack outlet
        elif histogram_type is HistogramType.PACK_COOLANT_TEMP_OUTLET:
            if "hv_battery_temp_outlet" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            # Filter out sensor errors
            hv_battery_temp_outlet = [dp for dp in self.can_data["hv_battery_temp_outlet"] if -40 < dp.value <= 100]

            hv_battery_temp_outlet_hist = self._create_custom_histogram(data_type=self.value_id_dict[1273],
                                                                        data=hv_battery_temp_outlet,
                                                                        bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                            histogram_type][0])
            if plot:
                hv_battery_temp_outlet_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(hv_battery_temp_outlet_hist)
            logging.info("--> Calculated histogram of coolant temperature at battery pack outlet successfully.")

        # rear rotor temperature
        elif histogram_type is HistogramType.ROTOR_REAR_TEMP:
            if "rear_motor_rotor_temp" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            rear_motor_rotor_temp_hist = self._create_histogram(data_type_name="rear_motor_rotor_temp",
                                                                bins=self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])
            if plot:
                rear_motor_rotor_temp_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(rear_motor_rotor_temp_hist)
            logging.info("--> Calculated histogram of temperatur of rear motor rotor successfully.")

        # rear stator temperature
        elif histogram_type is HistogramType.STATOR_REAR_TEMP:
            if "temp_rear_motor_stator" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            temp_rear_motor_stator_hist = self._create_histogram(data_type_name="temp_rear_motor_stator",
                                                                 bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                     histogram_type][0])
            if plot:
                temp_rear_motor_stator_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(temp_rear_motor_stator_hist)
            logging.info("--> Calculated histogram of temperatur of rear motor stator successfully.")

        # interior temperatur
        elif histogram_type is HistogramType.INTERIOR_TEMP:
            if "interior_temp" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            interior_temp_hist = self._create_histogram(data_type_name="interior_temp",
                                                        bins=self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])
            if plot:
                interior_temp_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(interior_temp_hist)
            logging.info("--> Calculated histogram of interior temperatur successfully.")

        # ptc1 current
        elif histogram_type is HistogramType.PTC1_CURRENT:
            if "ptc1_current" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            ptc1_current_hist = self._create_histogram(data_type_name="ptc1_current",
                                                       bins=self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])
            if plot:
                ptc1_current_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(ptc1_current_hist)
            logging.info("--> Calculated histogram of PTC 1 current successfully.")

        # ptc2 current
        elif histogram_type is HistogramType.PTC2_CURRENT:
            if "ptc2_current" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            ptc2_current_hist = self._create_histogram(data_type_name="ptc2_current",
                                                       bins=self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])
            if plot:
                ptc2_current_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(ptc2_current_hist)
            logging.info("--> Calculated histogram of PTC 2 current successfully.")

        # ptc voltage
        elif histogram_type is HistogramType.PTC_VOLTAGE:
            if "ptc_voltage" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            ptc1_current_hist = self._create_histogram(data_type_name="ptc_voltage",
                                                       bins=self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])
            if plot:
                ptc1_current_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(ptc1_current_hist)
            logging.info("--> Calculated histogram of PTC voltage successfully.")

        # rear inverter temperatur
        elif histogram_type is HistogramType.REAR_INVERTER_TEMP:
            if "temp_rear_inverter" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            # Filter out sensor errors
            temp_rear_inverter_inlet = [dp for dp in self.can_data["temp_rear_inverter"] if -20 < dp.value <= 100]

            temp_rear_inverter_inlet_hist = self._create_custom_histogram(data_type=self.value_id_dict[963],
                                                                          data=temp_rear_inverter_inlet,
                                                                          bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                              histogram_type][0])
            if plot:
                temp_rear_inverter_inlet_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(temp_rear_inverter_inlet_hist)
            logging.info("--> Calculated histogram of temperature of the rear inverter successfully.")

        # traveled distance
        elif histogram_type is HistogramType.TRAVELED_DISTANCE:
            if self.traveled_distance is None:
                raise RawDataError("Missing data for calculation!")

            traveled_distance = Datapoint("traveled_distance",
                                          self.traveled_distance,
                                          datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S"))
            traveled_distance_hist = self._create_custom_histogram(data_type=self.value_id_dict[1299],
                                                                   data=[traveled_distance],
                                                                   bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                       histogram_type][0])
            if plot:
                traveled_distance_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(traveled_distance_hist)
            logging.info("--> Calculated histogram of traveled distance successfully.")

        # power consumption auxiliaries
        elif histogram_type is HistogramType.AUXILIARIES_POWER:
            if "hv_aux_power" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            # Filter out sensor errors
            power_auxiliaries = [dp for dp in self.can_data["hv_aux_power"] if 0 < dp.value <= 20000]

            power_auxiliaries_hist = self._create_custom_histogram(data_type=self.value_id_dict[56],
                                                                   data=power_auxiliaries,
                                                                   bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                       histogram_type][0])
            if plot:
                power_auxiliaries_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(power_auxiliaries_hist)
            logging.info("--> Calculated histogram of auxiliaries power consumption successfully.")

        # ptc1 power
        elif histogram_type is HistogramType.PTC1_POWER:
            if "ptc1_current" not in self.can_data.keys() or "ptc_voltage" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            ptc1_voltage = self.can_data["ptc_voltage"]
            ptc1_current = self.can_data["ptc1_current"]

            # Use timeseries to match correct voltage and current values
            data = [ptc1_voltage, ptc1_current]
            latest_first_dp_timestamp = max([pack_temp_data_i[0].timestamp for pack_temp_data_i in data])
            earliest_last_dp_timestamp = min([pack_temp_data_i[-1].timestamp for pack_temp_data_i in data])
            if earliest_last_dp_timestamp < latest_first_dp_timestamp:
                raise RawDataError("Not enough data for calculations!")
            timeseries = np.arange(start=latest_first_dp_timestamp, stop=earliest_last_dp_timestamp,
                                   step=timedelta(milliseconds=250)).astype(datetime)
            timeseries = Track._add_data_gaps_to_timeseries(self.can_data["hv_battery_current"], timeseries)

            # Match data to timeseries
            data_in_timeseries = []
            for i in range(len(data)):
                data_in_timeseries.append(
                    Track._match_data_to_timeseries(data[i], timeseries))

            ptc1_power = []
            for i in range(len(data_in_timeseries[0])):
                voltage_i = data_in_timeseries[0][i]
                current_i = data_in_timeseries[1][i]

                if current_i.value >= 0:
                    ptc1_power.append(Datapoint(variable_name="ptc1_power",
                                                variable_value=float(voltage_i.value) * float(current_i.value),
                                                time_of_measurement=timeseries[i],
                                                road_type=current_i.road_type))
                else:
                    raise RawDataError("Negative current at ptc element not physically possible.")

            power_hist = self._create_custom_histogram(data_type=self.value_id_dict[1300],
                                                       data=ptc1_power,
                                                       bins=self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])
            if plot:
                power_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(power_hist)
            logging.info("--> Calculated histogram of ptc1 power successfully.")

        # ptc2 power
        elif histogram_type is HistogramType.PTC2_POWER:
            if "ptc2_current" not in self.can_data.keys() or "ptc_voltage" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            ptc2_voltage = self.can_data["ptc_voltage"]
            ptc2_current = self.can_data["ptc2_current"]

            # Use timeseries to match correct voltage and current values
            data = [ptc2_voltage, ptc2_current]
            latest_first_dp_timestamp = max([data_i[0].timestamp for data_i in data])
            earliest_last_dp_timestamp = min([data_i[-1].timestamp for data_i in data])
            if earliest_last_dp_timestamp < latest_first_dp_timestamp:
                raise RawDataError("Not enough data for calculations!")
            timeseries = np.arange(start=latest_first_dp_timestamp, stop=earliest_last_dp_timestamp,
                                   step=timedelta(milliseconds=250)).astype(datetime)
            timeseries = Track._add_data_gaps_to_timeseries(self.can_data["hv_battery_current"], timeseries)

            # Match data to timeseries
            data_in_timeseries = []
            for i in range(len(data)):
                data_in_timeseries.append(Track._match_data_to_timeseries(data[i], timeseries))

            ptc2_power = []
            for i in range(len(data_in_timeseries[0])):
                voltage_i = data_in_timeseries[0][i]
                current_i = data_in_timeseries[1][i]

                if current_i.value >= 0:
                    ptc2_power.append(Datapoint(variable_name="ptc2_power",
                                                variable_value=float(voltage_i.value) * float(current_i.value),
                                                time_of_measurement=timeseries[i],
                                                road_type=current_i.road_type))
                else:
                    raise RawDataError("Negative current at ptc element not physically possible.")

            power_hist = self._create_custom_histogram(data_type=self.value_id_dict[1301],
                                                       data=ptc2_power,
                                                       bins=self.BINS_FOR_HISTOGRAMTYPE[histogram_type][0])
            if plot:
                power_hist.plot(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(power_hist)
            logging.info("--> Calculated histogram of ptc2 power successfully.")

        # 2D C-Rate X max pack temp
        elif histogram_type is HistogramType.HIST2D_C_RATE__PACK_TEMP_MAX:
            if "hv_battery_current" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            # Prepare date for histogram
            c_rates = [Datapoint("cell_c_rate", float(i[0]) / self.vehicle.hv_cap, i[1]) for i in
                       [(dp.value, dp.timestamp) for dp in self.can_data["hv_battery_current"]]]

            # Check if temperature data is cached. If not, calculate.
            if "max_temp" in self.cache.keys():
                max_temp = self.cache["max_temp"]
            else:
                self.calculate_histogram(HistogramType.PACK_TEMP_MAX, keep_cache=True)
                self.histogram_list.pop()
                max_temp = self.cache["max_temp"]

            crate_packtempmax_2dhist = self._create_custom_2d_histogram(x_data_type=self.value_id_dict[1288],
                                                                        y_data_type=self.value_id_dict[1209],
                                                                        x_data=c_rates,
                                                                        y_data=max_temp,
                                                                        x_bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                            histogram_type][0],
                                                                        y_bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                            histogram_type][1])
            if plot:
                crate_packtempmax_2dhist.plot_2d(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
                crate_packtempmax_2dhist.plot_3d(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(crate_packtempmax_2dhist)
            logging.info("--> Calculated 2D histogram of c-rate and maximal pack temperature successfully.")

        # 2D C-Rate X pack temp delta
        elif histogram_type is HistogramType.HIST2D_C_RATE__PACK_TEMP_DELTA:
            if "hv_battery_current" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            # Prepare date for histogram
            c_rates = [Datapoint("cell_c_rate", float(i[0]) / self.vehicle.hv_cap, i[1]) for i in
                       [(dp.value, dp.timestamp) for dp in self.can_data["hv_battery_current"]]]

            # Check if temperature data is cached. If not, calculate.
            if "max_temp" in self.cache.keys() and "min_temp" in self.cache.keys():
                max_temp = self.cache["max_temp"]
                min_temp = self.cache["min_temp"]
            else:
                self.calculate_histogram(HistogramType.PACK_TEMP_DELTA, keep_cache=True)
                self.histogram_list.pop()
                max_temp = self.cache["max_temp"]
                min_temp = self.cache["min_temp"]

            crate_packtempdelta_2dhist = self._create_custom_2d_histogram(x_data_type=self.value_id_dict[1288],
                                                                          y_data_type=self.value_id_dict[1289],
                                                                          x_data=c_rates,
                                                                          y_data=list(
                                                                              np.subtract(max_temp, min_temp)),
                                                                          x_bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                              histogram_type][0],
                                                                          y_bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                              histogram_type][1])
            if plot:
                crate_packtempdelta_2dhist.plot_2d(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
                crate_packtempdelta_2dhist.plot_3d(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(crate_packtempdelta_2dhist)
            logging.info("--> Calculated 2D histogram of c-rate and pack temperature difference successfully.")

        # 2D C-Rate X soc
        elif histogram_type is HistogramType.HIST2D_C_RATE__PACK_SOC:
            if "hv_battery_current" not in self.can_data.keys() or "hv_soc" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            # Prepare date for histogram
            c_rates = [Datapoint("cell_c_rate", float(i[0]) / self.vehicle.hv_cap, i[1]) for i in
                       [(dp.value, dp.timestamp) for dp in self.can_data["hv_battery_current"]]]

            crate_soc_2dhist = self._create_custom_2d_histogram(x_data_type=self.value_id_dict[1288],
                                                                y_data_type=self.value_id_dict[900],
                                                                x_data=c_rates,
                                                                y_data=self.can_data["hv_soc"],
                                                                x_bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                    histogram_type][0],
                                                                y_bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                    histogram_type][1])
            if plot:
                crate_soc_2dhist.plot_2d(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
                crate_soc_2dhist.plot_3d(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(crate_soc_2dhist)
            logging.info("--> Calculated 2D histogram of c-rate and SOC successfully.")

        # 2D C-Rate X vehicle speed
        elif histogram_type is HistogramType.HIST2D_C_RATE__VEHICLE_SPEED:
            if "hv_battery_current" not in self.can_data.keys() or "vehicle_speed" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            # Prepare date for histogram
            c_rates = [Datapoint("cell_c_rate", float(i[0]) / self.vehicle.hv_cap, i[1]) for i in
                       [(dp.value, dp.timestamp) for dp in self.can_data["hv_battery_current"]]]

            crate_speed_2dhist = self._create_custom_2d_histogram(x_data_type=self.value_id_dict[1288],
                                                                  y_data_type=self.value_id_dict[4],
                                                                  x_data=c_rates,
                                                                  y_data=self.can_data["vehicle_speed"],
                                                                  x_bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                      histogram_type][0],
                                                                  y_bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                      histogram_type][1])
            if plot:
                crate_speed_2dhist.plot_2d(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
                crate_speed_2dhist.plot_3d(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(crate_speed_2dhist)
            logging.info("--> Calculated 2D histogram of c-rate and vehicle speed successfully.")

        # 2D pack temp max X soc
        elif histogram_type is HistogramType.HIST2D_PACK_TEMP_MAX__PACK_SOC:
            if "hv_soc" not in self.can_data.keys():
                raise RawDataError("Missing data for calculation!")

            # Prepare date for histogram
            # Check if temperature data is cached. If not, calculate.
            if "max_temp" in self.cache.keys():
                max_temp = self.cache["max_temp"]
            else:
                self.calculate_histogram(HistogramType.PACK_TEMP_MAX, keep_cache=True)
                self.histogram_list.pop()
                max_temp = self.cache["max_temp"]

            # Create histogram
            soc_packtempmax_2dhist = self._create_custom_2d_histogram(x_data_type=self.value_id_dict[1209],
                                                                      y_data_type=self.value_id_dict[900],
                                                                      x_data=max_temp,
                                                                      y_data=self.can_data["hv_soc"],
                                                                      x_bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                          histogram_type][0],
                                                                      y_bins=self.BINS_FOR_HISTOGRAMTYPE[
                                                                          histogram_type][1])
            if plot:
                soc_packtempmax_2dhist.plot_2d(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
                soc_packtempmax_2dhist.plot_3d(title_prefix + self.TITLE_FOR_HISTOGRAMTYPE[histogram_type])
            self.histogram_list.append(soc_packtempmax_2dhist)
            logging.info("--> Calculated 2D histogram of maximal pack temperature and SOC successfully.")

        else:
            raise NotImplementedError("Histogram of type '" + str(histogram_type.name) + "' is not implemented for "
                                                                                         "tracks.")

        if not keep_cache:
            self.cache = {}

    def remove_charging_from_data(self):
        """
        Removes charging processes from track data.
        """
        # Make sure, all data is available
        if len(self.value_id_dict.keys()) == 0:
            self.get_value_id_data()

        if len(self.can_data.keys()) == 0:
            self.fetch_can_data()

        if "hv_battery_current" not in self.can_data.keys() or "vehicle_speed" not in self.can_data.keys():
            raise ValueError("'hv_battery_current' and 'vehicle_speed' needed to detect charging processes. "
                             "Is the track complete?")

        # Match data to have comparable data
        data = [self.can_data["hv_battery_current"], self.can_data["vehicle_speed"]]
        latest_first_dp_timestamp = max([pack_temp_data_i[0].timestamp for pack_temp_data_i in data])
        earliest_last_dp_timestamp = min([pack_temp_data_i[-1].timestamp for pack_temp_data_i in data])
        timeseries = np.arange(start=latest_first_dp_timestamp, stop=earliest_last_dp_timestamp,
                               step=timedelta(milliseconds=250)).astype(datetime)
        timeseries = Track._add_data_gaps_to_timeseries(self.can_data["hv_battery_current"], timeseries)

        data_in_timeseries = []
        for i in range(len(data)):
            data_in_timeseries.append(
                Track._match_data_to_timeseries(data[i], timeseries))

        hv_current = data_in_timeseries[0]
        vehicle_speed = data_in_timeseries[1]
        del data_in_timeseries

        # Detect charging in track
        areas_to_remove = []
        start = None
        for i in range(len(timeseries)):
            if hv_current[i].value > 10 and vehicle_speed[i].value < 1:
                if start is None:
                    start = timeseries[i]
            elif start is not None:
                areas_to_remove.append((start, timeseries[i - 1]))
                start = None
        if start is not None:
            areas_to_remove.append((start, None))

        # Remove charging from data
        new_can_data = {}
        for key, value in self.can_data.items():
            new_data = []
            for dp in value:
                remove = False
                for area in areas_to_remove:
                    if area[1] is None:
                        if area[0] <= dp.timestamp:
                            remove = True
                            break
                    elif area[0] <= dp.timestamp < area[1]:
                        remove = True
                        break
                if not remove:
                    new_data.append(dp)
            new_can_data[key] = new_data

        # Save cropped data
        self.can_data = new_can_data

    def _check_if_already_processed(self, histogram_types: List[HistogramType]):
        """
        Checks, if the requested histograms are already processed.
        :param histogram_types: List, HistogramTypes of histograms that should be calculated.
        :return: List of HistogramTypes, that still need to be calculated.
        """
        conn = psycopg2.connect(database="mobtrack",
                                user=os.environ["db_user"],
                                password=os.environ["db_pwd"],
                                host=os.environ["host"],
                                port="5432")
        conn.autocommit = True
        cur = conn.cursor()

        # Get all already processed track ids
        cur.execute('''SELECT DISTINCT track_id FROM track.track_ev_histograms;''')
        processed_track_ids = cur.fetchall()

        # Get all already processed tracks
        sql = '''SELECT value_id_x, value_id_y, road_type, bins_id_x, bins_id_y FROM track.track_ev_histograms WHERE track_id=%s;'''
        sql_parameter = [self.id]
        cur.execute(sql, sql_parameter)
        calculated_histograms = cur.fetchall()

        if self.id not in np.array(processed_track_ids).flatten():
            return histogram_types

        remaining_histogram_types = []
        for hist_type in histogram_types:
            hist_val_id_x, hist_val_id_y = self.get_value_ids_for_histogram_type(hist_type)
            bins = self.BINS_FOR_HISTOGRAMTYPE[hist_type]

            already_calculated = False
            for calc_hist in calculated_histograms:
                # Check if histogram of requested type has been already calculated
                if calc_hist[0] == hist_val_id_x and calc_hist[1] == hist_val_id_y:
                    # TODO: Add road type check here before checking the bins... Not implemented right now.
                    # Check if bins are the same
                    sql = '''SELECT maximum, minimum, stepsize FROM track.histogram_bins WHERE id=%s;'''
                    sql_parameter = [calc_hist[3]]
                    cur.execute(sql, sql_parameter)
                    db_calc_bins_x = cur.fetchone()
                    calc_bins_x = Bins(maximum=db_calc_bins_x[0], minimum=db_calc_bins_x[1], step_size=db_calc_bins_x[2])

                    if calc_hist[4] == 0:
                        calc_bins_y = None
                    else:
                        sql = '''SELECT maximum, minimum, stepsize FROM track.histogram_bins WHERE id=%s;'''
                        sql_parameter = [calc_hist[4]]
                        cur.execute(sql, sql_parameter)
                        db_calc_bins_y = cur.fetchone()
                        calc_bins_y = Bins(maximum=db_calc_bins_y[0], minimum=db_calc_bins_y[1],
                                           step_size=db_calc_bins_y[2])

                    if calc_bins_x == bins[0] and calc_bins_y == bins[1]:
                        already_calculated = True
                        break

            if not already_calculated:
                remaining_histogram_types.append(hist_type)

        cur.close()
        conn.close()
        return remaining_histogram_types

    def analyze_peaks(self, data: List[Datapoint], split_lengths, min_prominences, overlap):
        """
        Returns amplitudes and frequencies of all peaks in data.
        """
        if len(split_lengths) != len(min_prominences):
            raise ValueError("'split_lengths' and 'min_prominences' must have the same length!")

        peaks = {}

        data_gaps = self._get_data_gaps(data, min_gap_len=1)

        # Loop through every split length
        for i in range(len(split_lengths)):
            split_length = split_lengths[i]
            min_promi = min_prominences[i]

            # Calculate the frequencies and amplitudes of the driving cycle by a sliding window approach
            start_index = 0
            while start_index < len(data) - split_length:
                current_split = data[start_index:(split_length + start_index)]

                idx, amplitudes, frequencies = self.analyze_split(start_index, current_split, min_promi, data_gaps)

                for j in range(len(idx)):
                    if idx[j] not in peaks.keys():
                        peaks[idx[j]] = {"amplitude": amplitudes[j], "frequency": frequencies[j]}

                start_index = int(start_index + split_length / overlap)

        return peaks

    def analyze_split(self, idx_offset, data_split, min_prominence, data_gaps):
        """
        Finds peaks, their amplitude and their frequency for a given data split.
        """
        # Extract local minima and maxima
        max_peaks, _ = scisig.find_peaks([dp.value for dp in data_split], prominence=min_prominence)
        min_peaks, _ = scisig.find_peaks([-dp.value for dp in data_split], prominence=min_prominence)

        idx = []
        amplitudes = []
        frequencies = []

        iter_len = min([len(min_peaks), len(max_peaks)])
        for i in range(0, iter_len - 1):
            max1_idx = max_peaks[i]
            max1_val = data_split[max1_idx]
            max2_idx = max_peaks[i + 1]
            max2_val = data_split[max2_idx]
            min1_idx = min_peaks[i]
            min1_val = data_split[min_peaks[i]]
            min2_idx = min_peaks[i + 1]
            min2_val = data_split[min_peaks[i + 1]]

            # Determine minimum between two maxima
            min_idx = min1_idx
            min_val = min1_val
            if min1_idx < max1_idx:
                min_idx = min2_idx
                min_val = min2_val

            # Determine maximum between two minima
            max_idx = max1_idx
            max_val = max1_val
            if max1_idx < min1_idx:
                max_idx = max2_idx
                max_val = max2_val

            # DISCHARGE PEAK
            if min_val.value < 0:
                # Check if data gap is between two peaks
                in_data_gap = False
                for gap in data_gaps:
                    if max2_val.timestamp >= gap[1] and max1_val.timestamp <= gap[0]:
                        in_data_gap = True
                        break

                time = (max2_val.timestamp - max1_val.timestamp).total_seconds()
                if in_data_gap:
                    logging.warning("--> Long peak duration ({}s) occurred due to data gap. Filtered out!".format(time))
                    continue

                # Amplitude is defined as value difference between first maximum % and the minimum
                amplitude = abs(max1_val.value - min_val.value)
                peak_idx = idx_offset + min_idx

                if peak_idx not in idx:
                    idx.append(peak_idx)
                    amplitudes.append(amplitude)
                    frequencies.append(1 / time)

            # CHARGE PEAK
            if max_val.value > 0:
                # Check if data gap is between two peaks
                in_data_gap = False
                for gap in data_gaps:
                    if min2_val.timestamp >= gap[1] and min1_val.timestamp <= gap[0]:
                        in_data_gap = True
                        break

                time = (min2_val.timestamp - min1_val.timestamp).total_seconds()
                if in_data_gap:
                    logging.warning("--> Long peak duration ({}s) occurred due to data gap. Filtered out!".format(time))
                    continue

                # Amplitude is defined as value difference between first maximum % and the minimum
                amplitude = abs(min1_val.value - max_val.value)
                peak_idx = idx_offset + max_idx

                if peak_idx not in idx:
                    idx.append(peak_idx)
                    amplitudes.append(amplitude)
                    frequencies.append(1 / time)

        return idx, amplitudes, frequencies,
