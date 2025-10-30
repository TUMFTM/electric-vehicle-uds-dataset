# Class Vehicle
# Describes a test vehicle
# Author: Lukas L. Köning
import os

import numpy as np
import psycopg2

from .bev_usage import BEVUsage, HistogramType
from .charging import Charging
from .datapoint import RoadType
from .datatype import DataType
from .histogram import Histogram, Histogram2D, Bins
from .track import Track


class Vehicle:
    def __init__(self, vehicle_id, vehicle_name, battery_capacity):
        self.id = vehicle_id
        self.name = vehicle_name
        self.hv_cap = battery_capacity

    def get_tracks(self):
        """
        Collects all tracks from this vehicle from the database.
        :return: list of tracks
        """
        tracks = []

        conn = psycopg2.connect(database="mobtrack",
                                user=os.environ["db_user"],
                                password=os.environ["db_pwd"],
                                host=os.environ["host"],
                                port="5432")
        conn.autocommit = True
        cur = conn.cursor()

        # Get all already processed tracks
        cur.execute('''SELECT track_id FROM track.track_ev_histograms;''')

        processed_tracks = cur.fetchall()

        # Filter out all tracks, that are shorter than 1min, 500m and have an average speed of less than 5 km/h
        sql = '''SELECT t.id, t.start_time, t.stop_time, tm.distance 
        FROM track.track t INNER JOIN track.track_metadata tm ON t.id=tm.track_id 
        WHERE EXTRACT(EPOCH FROM (t.stop_time - t.start_time))/60 > 1 
        AND t.vehicle_id=%s
        AND tm.distance > 500 and tm.avg_speed > 5/3.6'''
        sql_parameter = [self.id]
        cur.execute(sql, sql_parameter)

        sql_result = cur.fetchall()
        for track_data in sql_result:
            tracks.append(Track(track_id=track_data[0],
                                start=track_data[1].strftime("%Y-%m-%d %H:%M:%S"),
                                end=track_data[2].strftime("%Y-%m-%d %H:%M:%S"),
                                vehicle=self,
                                processed=track_data[0] in processed_tracks,
                                distance=track_data[3] / 1000))

        conn.close()

        # Sort tracks, so the first entry is the most recent track
        tracks.sort(key=lambda x: x.start_time, reverse=True)

        # Add previous end time to tracks
        for t in range(len(tracks) - 1):
            tracks[t].prior_usage_end = tracks[t + 1].end_time

        return tracks

    def get_charging_processes(self):
        """
        Collects all charging processes from this vehicle from the database.
        :return: list of charging objects
        """
        charging_processes = []

        conn = psycopg2.connect(database="mobtrack",
                                user=os.environ["db_user"],
                                password=os.environ["db_pwd"],
                                host=os.environ["host"],
                                port="5432")
        conn.autocommit = True
        cur = conn.cursor()

        # Get all already processed charging processes
        cur.execute('''SELECT charging_id FROM track.charging_ev_histograms;''')
        processed_charging_processes = cur.fetchall()

        # Filter out all charging processes, that are shorter than 1min, and have invalid soc data
        sql = '''SELECT id, start_time, stop_time, start_soc, stop_soc 
        FROM track.charging 
        WHERE vehicle_id=%s 
        AND EXTRACT(EPOCH FROM (stop_time - start_time))/60 > 1
        AND start_soc < 100 
        AND stop_soc > 0 
        AND stop_soc-start_soc > 0;'''
        sql_parameter = [self.id]
        cur.execute(sql, sql_parameter)

        sql_result = cur.fetchall()
        for charging_data in sql_result:
            charging_processes.append(Charging(charging_id=charging_data[0],
                                               start=charging_data[1].strftime("%Y-%m-%d %H:%M:%S"),
                                               end=charging_data[2].strftime("%Y-%m-%d %H:%M:%S"),
                                               vehicle=self,
                                               processed=charging_data[0] in processed_charging_processes))

        conn.close()

        # Sort charging processes, so the first entry is the most recent charging process
        charging_processes.sort(key=lambda x: x.start_time, reverse=True)

        return charging_processes

    def get_track_histograms_from_database(self, track_id=None, hist_type: HistogramType = None):
        """
        Fetches histogram data from the database for all tracks or the one specified
        :param track_id: track_id for which all histograms should be fetched
        :param hist_type: HistogramType, use this if you want only a specific histogram type
        :return: list of histogram objects, list of corresponding bin_ids
        """
        # Get some needed data
        value_id_x, value_id_y = BEVUsage.get_value_ids_for_histogram_type(histogram_type=hist_type)
        track_ids = [track.id for track in self.get_tracks()]

        # Initialize some variables
        value_id_dict = {}
        bins_dict = {}
        histograms = []
        bin_ids = []

        # Open connection to database
        conn = psycopg2.connect(database="mobtrack",
                                user=os.environ["db_user"],
                                password=os.environ["db_pwd"],
                                host=os.environ["host"],
                                port="5432")
        conn.autocommit = True
        cur = conn.cursor()

        # Get value_id data
        cur.execute('''SELECT value_id, variable_name, name_de, unit FROM vehicle.can_value''')

        sql_result = cur.fetchall()
        for value_data in sql_result:
            value_id_dict[value_data[0]] = DataType(value_data[0], value_data[1], value_data[2], value_data[3])

        # Get bin data from database
        cur.execute('''SELECT id, maximum, minimum, stepsize FROM track.histogram_bins''')
        sql_result = cur.fetchall()
        for b in sql_result:
            bins_dict[b[0]] = Bins(maximum=b[1], minimum=b[2], step_size=b[3])

        # Get histograms for all tracks or the specified one
        if track_id is None:
            if hist_type is None:
                cur.execute('''SELECT value_id_x, value_id_y, counts, bins_id_x, road_type, track_id, bins_id_y
                FROM track.track_ev_histograms''')
            else:
                sql = '''SELECT value_id_x, value_id_y, counts, bins_id_x, road_type, track_id, bins_id_y 
                               FROM track.track_ev_histograms
                               WHERE value_id_x=%s AND value_id_y=%s'''
                sql_parameter = [value_id_x, value_id_y]
                cur.execute(sql, sql_parameter)
        else:
            if hist_type is None:
                sql = '''SELECT value_id_x, value_id_y, counts, bins_id_x, road_type, track_id, bins_id_y
                FROM track.track_ev_histograms 
                WHERE track_id=%s'''
                sql_parameter = [track_id]
                cur.execute(sql, sql_parameter)
            else:
                sql = '''SELECT value_id_x, value_id_y, counts, bins_id_x, road_type, track_id, bins_id_y 
                               FROM track.track_ev_histograms 
                               WHERE track_id=%s 
                               AND value_id_x=%s AND value_id_y=%s'''
                sql_parameter = [track_id, value_id_x, value_id_y]
                cur.execute(sql, sql_parameter)

        # Convert data from database to histogram objects
        histograms_data = cur.fetchall()
        for hist_data in histograms_data:
            # Only process tracks from this vehicle
            if hist_data[5] not in track_ids:
                continue

            if hist_data[1] == 0:
                bin_ids.append(hist_data[3])
                histograms.append(Histogram(data_type=value_id_dict[hist_data[0]],
                                            road_type=RoadType[hist_data[4].upper()],
                                            counts=np.array(hist_data[2]),
                                            bins=bins_dict[hist_data[3]]))
            else:
                bin_ids.append(hist_data[3] + 100000 * hist_data[6])
                histograms.append(Histogram2D(x_data_type=value_id_dict[hist_data[0]],
                                              y_data_type=value_id_dict[hist_data[1]],
                                              counts=np.array(hist_data[2]),
                                              x_bins=bins_dict[hist_data[3]],
                                              y_bins=bins_dict[hist_data[6]],
                                              road_type=RoadType[hist_data[4].upper()]))

        cur.close()
        conn.close()

        return histograms, bin_ids

    def get_charging_histograms_from_database(self, charging_id=None, hist_type: HistogramType = None):
        """
        Fetches histogram data from the database for all charging processes or the one specified
        :param charging_id: charging_id for which all histograms should be fetched
        :param hist_type: HistogramType, use this if you want only a specific histogram type
        :return: list of histogram objects, list of corresponding bin_ids
        """
        # Get some needed data
        value_id_x, value_id_y = BEVUsage.get_value_ids_for_histogram_type(histogram_type=hist_type)
        charging_ids = [chg.id for chg in self.get_charging_processes()]

        # Initialize some variables
        value_id_dict = {}
        bins_dict = {}
        histograms = []
        bin_ids = []

        # Open connection to database
        conn = psycopg2.connect(database="mobtrack",
                                user=os.environ["db_user"],
                                password=os.environ["db_pwd"],
                                host=os.environ["host"],
                                port="5432")
        conn.autocommit = True
        cur = conn.cursor()

        # Get value_id data
        cur.execute('''SELECT value_id, variable_name, name_de, unit FROM vehicle.can_value''')

        sql_result = cur.fetchall()
        for value_data in sql_result:
            value_id_dict[value_data[0]] = DataType(value_data[0], value_data[1], value_data[2], value_data[3])

        # Get bin data from database
        cur.execute('''SELECT id, maximum, minimum, stepsize FROM track.histogram_bins''')
        sql_result = cur.fetchall()
        for b in sql_result:
            bins_dict[b[0]] = Bins(maximum=b[1], minimum=b[2], step_size=b[3])

        # Get histograms for all tracks or the specified one
        if charging_id is None:
            if hist_type is None:
                cur.execute('''SELECT value_id_x, value_id_y, counts, bins_id_x, road_type, charging_id, bins_id_y 
                FROM track.charging_ev_histograms''')
            else:
                sql = '''SELECT value_id_x, value_id_y, counts, bins_id_x, road_type, charging_id, bins_id_y 
                               FROM track.charging_ev_histograms
                               WHERE value_id_x=%s AND value_id_y=%s'''
                sql_parameter = [value_id_x, value_id_y]
                cur.execute(sql, sql_parameter)
        else:
            if hist_type is None:
                sql = '''SELECT value_id_x, value_id_y, counts, bins_id_x, road_type, charging_id, bins_id_y 
                FROM track.charging_ev_histograms 
                WHERE charging_id=%s'''
                sql_parameter = [charging_id]
                cur.execute(sql, sql_parameter)
            else:
                sql = '''SELECT value_id_x, value_id_y, counts, bins_id_x, road_type, charging_id, bins_id_y 
                               FROM track.charging_ev_histograms 
                               WHERE charging_id=%s AND value_id_x=%s AND value_id_y=%s'''
                sql_parameter = [charging_id, value_id_x, value_id_y]
                cur.execute(sql, sql_parameter)

        # Convert data from database to histogram objects
        histograms_data = cur.fetchall()
        for hist_data in histograms_data:
            # Only process tracks from this vehicle
            if hist_data[5] not in charging_ids:
                continue

            if hist_data[1] == 0:
                bin_ids.append(hist_data[3])
                histograms.append(Histogram(data_type=value_id_dict[hist_data[0]],
                                            road_type=RoadType[hist_data[4].upper()],
                                            counts=np.array(hist_data[2]),
                                            bins=bins_dict[hist_data[3]]))
            else:
                bin_ids.append(hist_data[3] + 100000 * hist_data[6])
                histograms.append(Histogram2D(x_data_type=value_id_dict[hist_data[0]],
                                              y_data_type=value_id_dict[hist_data[1]],
                                              counts=np.array(hist_data[2]),
                                              x_bins=bins_dict[hist_data[3]],
                                              y_bins=bins_dict[hist_data[6]],
                                              road_type=RoadType[hist_data[4].upper()]))

        return histograms, bin_ids
