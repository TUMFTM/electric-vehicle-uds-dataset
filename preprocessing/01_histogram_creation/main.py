import logging
import os
import time
from copy import deepcopy
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import psycopg2
from tqdm import tqdm

from classes.bev_usage import BEVUsage, HistogramType
from classes.charging import Charging
from classes.datapoint import RoadType
from classes.datatype import DataType
from classes.exceptions import RawDataError
from classes.histogram import Histogram, Histogram2D, Bins
from classes.track import Track
from classes.vehicle import Vehicle


def process_new_tracks(vehicle: Vehicle, desired_track_histograms, desired_charging_histograms):
    """
    Function that should be called regularly, to process newly recorded bev usages for the given vehicle
    :param vehicle: Vehicle object, vehicle for that new bev usages should be processed
    :param desired_track_histograms: List of HistogramType objects, desired histogram types for tracks
    :param desired_charging_histograms: List of HistogramType objects, desired histogram types for charging processes
    :return:
    """
    logging.info("============================================================")
    logging.info("Processing vehicle '" + str(vehicle.name) + "'")
    logging.info("============================================================")
    try:
        # Get all charging processes and tracks from the database
        charging_processes = []
        tracks = []
        if len(desired_track_histograms) > 0:
            charging_processes = vehicle.get_tracks()
        if len(desired_charging_histograms) > 0:
            tracks = vehicle.get_charging_processes()
        bev_usages = charging_processes + tracks

        # Sort by timestamp
        bev_usages.sort(key=lambda x: datetime.strptime(x.start_time, "%Y-%m-%d %H:%M:%S"), reverse=False)

        # Add data for idle time calculation
        for i in range(1, len(bev_usages)):
            bev_usages[i].prior_usage_end = bev_usages[i - 1].end_time

        evaluated_distance = 0
        for i in range(len(bev_usages)):
            logging.info("Processing usage " + str(i + 1) + "/" + str(len(bev_usages)) + "...")
            usage = deepcopy(bev_usages[i])
            try:
                # Analyze each usage
                if type(usage) is Track:
                    usage.analyze(desired_histograms=desired_track_histograms, plot=False)
                    evaluated_distance += usage.traveled_distance
                elif type(usage) is Charging:
                    usage.analyze(desired_histograms=desired_charging_histograms, plot=False)

                # Save usage histograms to database
                usage.save_histograms_to_database(force_overwrite=False)
                logging.info("--> Saved calculated histograms to database successfully.")
            except (RawDataError, IndexError) as err:
                logging.warning(str(type(usage).__name__) + " could not be analyzed! Reason: " + str(err))
                

            # Clean up memory
            del usage

        logging.info("Evaluated data from " + str(evaluated_distance) + " km.")
    except Exception as err:
        logging.error("Processing of vehicle failed! " + str(type(err).__name__) + ": " + str(err))


def get_combined_histogram(vehicle: Vehicle, hist_type, charging=False):
    """
    Calculates and returns the combined histogram of given type for a defined vehicle
    :param vehicle: Vehicle object, vehicle for which the histograms should be combined
    :param hist_type: HistogramType, histogram type of the histograms that should be combined
    :param charging: Bool, calculate for charging processes or tracks?
    :return: dictionary, Histogram or Histogram2D object of all combined histograms grouped by bin_ids
    """
    # Get histograms from database
    if charging:
        calculated_histograms, bin_ids = vehicle.get_charging_histograms_from_database(hist_type=hist_type)
    else:
        calculated_histograms, bin_ids = vehicle.get_track_histograms_from_database(hist_type=hist_type)

    # Sum all histograms
    summed_histograms = {}
    for i in range(len(calculated_histograms)):
        if bin_ids[i] not in summed_histograms:
            summed_histograms[bin_ids[i]] = calculated_histograms[i]
        else:
            summed_histograms[bin_ids[i]] = summed_histograms[bin_ids[i]] + calculated_histograms[i]

    return summed_histograms


def postprocess_vehicle(vehicle: Vehicle, track_hist_types, charging_hist_types):
    """
    Postprocess the results from vehicle.
    :param vehicle: Vehicle object, vehicle to postprocess
    """
    logging.info("============================================================")
    logging.info("Postprocessing vehicle '" + str(vehicle.name) + "'")
    logging.info("============================================================")

    # Check, if output folder for vehicle exists
    if not os.path.isdir(os.path.join("output", "V" + str(vehicle.id))):
        os.mkdir(os.path.join("output", "V" + str(vehicle.id)))

    for track_hist_type in (pbar := tqdm(track_hist_types)):
        pbar.set_description("Postprocessing track histograms")
        summed_histograms = get_combined_histogram(vehicle=vehicle, hist_type=track_hist_type, charging=False)
        if len(summed_histograms.keys()) == 0:
            logging.info("No data found for histogram type '" + str(track_hist_type.name) + "'.")
            continue
        if len(summed_histograms.keys()) > 1:
            # TODO: Find a way to save for multiple bin ids
            logging.warning("Multiple bins for histogram type '" + str(track_hist_type.name)
                            + "'. Only one bin type can be processed right now. The others will be omitted.")
        for bin_id, histogram in summed_histograms.items():
            if type(histogram) is Histogram:
                # TODO: Add title to plot. Maybe global dictionary corresponding to histogramtype?
                # TODO: Add special postprocessing for some types
                # Special postprocessing
                if track_hist_type is HistogramType.IDLE_PERIOD_DURATION:
                    # Omit high peak with idle times below 15 Minutes
                    histogram.bins = Bins(minimum=histogram.bins.minimum + histogram.bins.step_size,
                                          maximum=histogram.bins.maximum, step_size=histogram.bins.step_size)
                    histogram.counts = histogram.counts[1:]
                    histogram.plot(save_dir=os.path.join("V" + str(vehicle.id), "tracks"))
                    plt.close()
                elif track_hist_type is HistogramType.CELL_C_RATE:
                    # Variable preparation
                    discharge_bin_indices = [i for i in range(len(histogram.bins) - 1) if
                                             histogram.bins.get_array()[i] <= 0]
                    charge_bin_indices = [i for i in range(len(histogram.bins) - 1) if
                                          histogram.bins.get_array()[i] > 0]
                    discharge_counts = [histogram.counts[i] for i in range(len(histogram.bins) - 1) if
                                        histogram.bins.get_array()[i] <= 0]
                    charge_counts = [histogram.counts[i] for i in range(len(histogram.bins) - 1) if
                                     histogram.bins.get_array()[i] > 0]

                    # Percentile calculation for discharge direction
                    relative_discharge_cumsum = np.cumsum(np.flip(discharge_counts)) / np.sum(discharge_counts) * 100
                    discharge_80th_percentile_bin_idx = len(relative_discharge_cumsum[relative_discharge_cumsum <= 80])
                    discharge_50th_percentile_bin_idx = len(relative_discharge_cumsum[relative_discharge_cumsum <= 50])
                    discharge_80th_percentile = np.around(
                        histogram.bins.get_array()[np.flip(discharge_bin_indices)[discharge_80th_percentile_bin_idx]],
                        2)
                    discharge_50th_percentile = np.around(
                        histogram.bins.get_array()[np.flip(discharge_bin_indices)[discharge_50th_percentile_bin_idx]],
                        2)
                    logging.info("--> [C-Rate][Entladerichtung] 80. Perzentil: " + str(discharge_80th_percentile))
                    logging.info(
                        "--> [C-Rate][Entladerichtung] 50. Perzentil (Median): " + str(discharge_50th_percentile))

                    # Percentile calculation for charge direction
                    relative_charge_cumsum = np.cumsum(charge_counts) / np.sum(charge_counts) * 100
                    charge_80th_percentile_bin_idx = len(relative_charge_cumsum[relative_charge_cumsum <= 80])
                    charge_50th_percentile_bin_idx = len(relative_charge_cumsum[relative_charge_cumsum <= 50])
                    charge_80th_percentile = np.around(
                        histogram.bins.get_array()[charge_bin_indices[charge_80th_percentile_bin_idx]], 2)
                    charge_50th_percentile = np.around(
                        histogram.bins.get_array()[charge_bin_indices[charge_50th_percentile_bin_idx]], 2)
                    logging.info("--> [C-Rate][Laderichtung] 80. Perzentil: " + str(charge_80th_percentile))
                    logging.info("--> [C-Rate][Laderichtung] 50. Perzentil (Median): " + str(charge_50th_percentile))

                    # Plotting
                    histogram.plot(show=False)
                    # TODO: Improve plotting?
                    plt.vlines(discharge_80th_percentile, 0, max(histogram.counts))
                    plt.vlines(charge_80th_percentile, 0, max(histogram.counts))

                    if histogram.road_type is RoadType.UNKNOWN:
                        plt.title("[Fahren] " + BEVUsage.TITLE_FOR_HISTOGRAMTYPE[track_hist_type])
                    else:
                        plt.title("[" + histogram.road_type.name.lower() + "]: " + "[Fahren] " +
                                  BEVUsage.TITLE_FOR_HISTOGRAMTYPE[track_hist_type])

                    plt.show(block=False)
                    if not os.path.isdir(os.path.join("output", "V" + str(vehicle.id), "tracks")):
                        os.mkdir(os.path.join("output", "V" + str(vehicle.id), "tracks"))
                    plt.savefig(os.path.join("output", "V" + str(vehicle.id), "tracks",
                                             "1DHistogram_" + histogram.data_type.var_name + ".png"), dpi=166)
                    plt.close()
                else:
                    histogram.plot(title="[Fahren] " + BEVUsage.TITLE_FOR_HISTOGRAMTYPE[track_hist_type],
                                   save_dir=os.path.join("V" + str(vehicle.id), "tracks"))
                    plt.close()
            elif type(histogram) is Histogram2D:
                histogram.plot_2d(title="[Fahren] " + BEVUsage.TITLE_FOR_HISTOGRAMTYPE[track_hist_type],
                                  save_dir=os.path.join("V" + str(vehicle.id), "tracks"))
                plt.close()
                histogram.plot_3d(title="[Fahren] " + BEVUsage.TITLE_FOR_HISTOGRAMTYPE[track_hist_type],
                                  save_dir=os.path.join("V" + str(vehicle.id), "tracks"))
                plt.close()

            histogram.to_pickle(os.path.join("V" + str(vehicle.id), "tracks", "pickle_files"))

    for charging_hist_type in (pbar := tqdm(charging_hist_types)):
        pbar.set_description("Postprocessing charging histograms")
        summed_histograms = get_combined_histogram(vehicle=vehicle, hist_type=charging_hist_type, charging=True)
        if len(summed_histograms.keys()) > 1:
            # TODO: Find a way to save for multiple bin ids
            logging.warning("Multiple bins for histogram type '" + str(charging_hist_type.name)
                            + "'. Only one bin type can be processed right now. The others will be omitted.")
        for bin_id, histogram in summed_histograms.items():
            if type(histogram) is Histogram:
                # TODO: Add title to plot. Maybe global dictionary corresponding to histogramtype?
                # TODO: Add special postprocessing for some types
                histogram.plot(title="[Laden] " + BEVUsage.TITLE_FOR_HISTOGRAMTYPE[charging_hist_type],
                               save_dir=os.path.join("V" + str(vehicle.id), "charging"))
                plt.close()
            elif type(histogram) is Histogram2D:
                histogram.plot_2d(title="[Laden] " + BEVUsage.TITLE_FOR_HISTOGRAMTYPE[charging_hist_type],
                                  save_dir=os.path.join("V" + str(vehicle.id), "charging"))
                plt.close()
                histogram.plot_3d(title="[Laden] " + BEVUsage.TITLE_FOR_HISTOGRAMTYPE[charging_hist_type],
                                  save_dir=os.path.join("V" + str(vehicle.id), "charging"))
                plt.close()
            histogram.to_pickle(os.path.join("V" + str(vehicle.id), "charging", "pickle_files"))


def get_histograms_from_database(track_id=None):
    """
    Fetches histogram data from the database for all tracks or the one specified
    :param track_id: track_id for which all histograms should be fetched
    :return: list of histogram objects
    """
    value_id_dict = {}
    bins_dict = {}
    histograms = []
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
    bins_result = cur.fetchall()
    existing_bins_dict = {}
    for b in bins_result:
        existing_bins_dict[b[0]] = Bins(maximum=b[1], minimum=b[2], step_size=b[3])

    # Get histograms for all tracks or the specified one
    if track_id is None:
        cur.execute('''SELECT value_id_x, value_id_y, counts, bins_id_x, road_type, bins_id_y
        FROM track.track_ev_histograms''')
    else:
        sql = '''SELECT value_id_x, value_id_y, counts, bins_id_x, road_type, bins_id_y 
        FROM track.track_ev_histograms 
        WHERE track_id=%s'''
        sql_parameters = [track_id]
        cur.execute(sql, sql_parameters)

    # Convert data from database to histogram objects
    histograms_data = cur.fetchall()
    for hist_data in histograms_data:
        if hist_data[1] == 0:
            histograms.append(Histogram(data_type=value_id_dict[hist_data[0]],
                                        road_type=RoadType[hist_data[4].upper()],
                                        counts=np.array(hist_data[2]),
                                        bins=bins_dict[hist_data[3]]))
        else:
            histograms.append(Histogram2D(x_data_type=value_id_dict[hist_data[0]],
                                          y_data_type=value_id_dict[hist_data[1]],
                                          counts=np.array(hist_data[2]),
                                          x_bins=bins_dict[hist_data[3]],
                                          y_bins=bins_dict[hist_data[5]],
                                          road_type=RoadType[hist_data[4].upper()]))

    return histograms


if __name__ == '__main__':
    log_path = os.environ["log_path"]
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s][%(levelname)s]: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[
                            logging.FileHandler(log_path + "vehicle_data_processing.log"),
                            logging.FileHandler("logs/vehicle_data_processing.log"),
                            logging.StreamHandler()
                        ]
                        )

    logging.info("Starting up")

    tesla_cap = 161  # Ah
    id3_cap = 78 * 2  # Ah
    seat_mii_cap = 60 * 2  # Ah

    ftmTesla3 = Vehicle("1700000394", "FTM Tesla Model 3", tesla_cap)
    ftmID3 = Vehicle("1700000296", "FTM VW ID.3", id3_cap)
    georgID3 = Vehicle("1700000635", "Georg's VW ID.3", id3_cap)
    seatMiiElectric = Vehicle("1700000775", "Seat Mii Electric", seat_mii_cap)

    cupra_349 = Vehicle("1700000988", "CS Cupra 349", id3_cap)
    cupra_213 = Vehicle("1700000989", "CS Cupra 213", id3_cap)
    cupra_288 = Vehicle("1700000990", "CS Cupra 288", id3_cap)
    cupra_397 = Vehicle("1700000991", "CS Cupra 397", id3_cap)
    cupra_204 = Vehicle("1700000992", "CS Cupra 204", id3_cap)

    # WORKFLOW
    start_time = time.time()

    vehicle_list = [seatMiiElectric, ftmTesla3, ftmID3, georgID3, cupra_349, cupra_213, cupra_288, cupra_397, cupra_204]
    vehicle_list = [ftmID3]

    logging.info("Using env variables: user {}, password {}, host {}".format(os.environ['db_user'],
                                                                             bool(os.environ['db_pwd']) * '*********',
                                                                             os.environ['host']))
    logging.info('Configured vehicles: {}'.format(vehicle_list))

    for v in vehicle_list:
        process_new_tracks(v, Track.IMPLEMENTED_HISTOGRAMS, Charging.IMPLEMENTED_HISTOGRAMS)
    end_time = time.time()
    dt = end_time - start_time
    logging.info("Processed {} vehicles. This took {} hours, {} minutes and {} seconds.".format(len(vehicle_list),
                                                                                                round(dt / 3600),
                                                                                                round((dt % 3600) / 60),
                                                                                                np.around(
                                                                                                    (dt % 3600) % 60,
                                                                                                    3)))
    # END WORKFLOW

    # POSTPROCESSING

    postprocess_vehicle(seatMiiElectric, Track.IMPLEMENTED_HISTOGRAMS, [])
    postprocess_vehicle(ftmTesla3, Track.IMPLEMENTED_HISTOGRAMS, [])
    postprocess_vehicle(ftmID3, Track.IMPLEMENTED_HISTOGRAMS, [])
    postprocess_vehicle(georgID3, Track.IMPLEMENTED_HISTOGRAMS, [])
    postprocess_vehicle(cupra_349, Track.IMPLEMENTED_HISTOGRAMS, [])
    postprocess_vehicle(cupra_213, Track.IMPLEMENTED_HISTOGRAMS, [])
    postprocess_vehicle(cupra_288, Track.IMPLEMENTED_HISTOGRAMS, [])
    postprocess_vehicle(cupra_397, Track.IMPLEMENTED_HISTOGRAMS, [])
    postprocess_vehicle(cupra_204, Track.IMPLEMENTED_HISTOGRAMS, [])

    # END POSTPROCESSING

    # # DEBUGGING
    # test_charging_process = [c for c in georgID3.get_charging_processes() if c.id == 2823][0]
    # # test_charging_process.analyze([HistogramType.HIST2D_C_RATE__PACK_TEMP_MAX])
    # # test_charging_process.analyze(Charging.IMPLEMENTED_HISTOGRAMS)
    # # test_charging_process.save_histograms_to_database()
    #
    # test_track = [t for t in georgID3.get_tracks() if t.id == 1500000496344][0]
    # test_track.analyze([HistogramType.HIST2D_C_RATE__PACK_SOC], plot=False)
    # test_track.analyze(Track.IMPLEMENTED_HISTOGRAMS, plot=False)
    # test_track.save_histograms_to_database()
    # # test_track.histogram_list[0].plot()
    # # END DEBUGGING
