# Class Histogram
# Contains all data of a histogram
# ASSUMPTION: Data is summed by time (seconds) with a value!
# Author: Lukas L. Köning
import calendar
import logging
import multiprocessing as mp
import os
import pickle
import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from config import THREADS
from .datapoint import Datapoint, RoadType
from .exceptions import HistogramError

# Some histograms are track-wise and have therefore no duration, they count per definition as one
# and must be evaluated correctly in the postprocessing!
TRACK_WISE_VARIABLES = [1290, 1291, 1292, 1296, 1298, 1299, 1302, 1303]


class Bins:

    def __init__(self, minimum, maximum, step_size):
        self.minimum = minimum
        self.maximum = maximum
        self.step_size = step_size

    def __eq__(self, other):
        if type(other) == Bins:
            return self.minimum == other.minimum and self.maximum == other.maximum and self.step_size == other.step_size
        elif type(other) == list:
            o_minimum = min(other)
            o_maximum = max(other)
            o_step_size = sorted(other)[1] - sorted(other)[0]
            return self.minimum == o_minimum and self.maximum == o_maximum and self.step_size == o_step_size
        else:
            raise TypeError("Bins can only be compared with other Bins objects or lists!")

    def get_array(self) -> list:
        """
        Returns the bins as list for further usage
        """
        return_object = np.arange(self.minimum, self.maximum + self.step_size, self.step_size).tolist()
        if type(return_object) is list:
            return return_object
        else:
            raise ValueError("Bin array is empty! Check maximum, minimum and step size!")

    def __len__(self):
        return len(self.get_array())


class Histogram:
    def __init__(self, data_type, road_type: RoadType, counts=None, bins: Bins = None):
        """
        Initialization function of a Histogram object.
        :param data_type: DataType object describing the data in the histogram
        :param road_type: RoadType value, categorizing the histogram according to the road-type
        :param counts: optional, list with precalculated counts
        :param bins: optional, list with precalculated bins
        """
        self.data_type = data_type
        self.counts = counts
        self.bins = bins
        self.road_type = road_type

    def __add__(self, other):
        if not type(other) is Histogram:
            raise TypeError("You can only sum two Histogram objects!")
        if not self.data_type == other.data_type:
            raise ArithmeticError("Histograms can only be combined, if type of data is the same.")
        if not self.bins == other.bins:
            raise ArithmeticError("Histograms can only be combined, if the bins are the same.")
        if not self.road_type == other.road_type:
            raise ArithmeticError("Histograms can only be combined, if road_type of data is the same.")
        return Histogram(data_type=self.data_type, counts=self.counts + other.counts, bins=self.bins,
                         road_type=self.road_type)

    def __sub__(self, other):
        if not type(other) is Histogram:
            raise TypeError("You can only subtract a Histogram object from a Histogram object!")
        if not self.data_type == other.data_type:
            raise ArithmeticError("Histograms can only be combined, if type of data is the same.")
        if not self.bins == other.bins:
            raise ArithmeticError("Histograms can only be combined, if the bins are the same.")
        if not self.road_type == other.road_type:
            raise ArithmeticError("Histograms can only be combined, if road_type of data is the same.")
        return Histogram(data_type=self.data_type, counts=self.counts - other.counts, bins=self.bins,
                         road_type=self.road_type)

    def __truediv__(self, other):
        if type(other) is float:
            return Histogram(data_type=self.data_type, counts=self.counts / other, bins=self.bins,
                             road_type=self.road_type)
        elif type(other) is Histogram:
            if not self.data_type == other.data_type:
                raise ArithmeticError("Histograms can only be combined, if type of data is the same.")
            if not self.bins == other.bins:
                raise ArithmeticError("Histograms can only be combined, if the bins are the same.")
            if not self.road_type == other.road_type:
                raise ArithmeticError("Histograms can only be combined, if road_type of data is the same.")
            return Histogram(data_type=self.data_type, counts=self.counts / other.counts, bins=self.bins,
                             road_type=self.road_type)
        else:
            raise TypeError("You can only divide a Histogram object by a float value or another Histogram object!")

    def __mul__(self, other):
        if not type(other) is float:
            raise TypeError("You can only multiply a Histogram object with a float value!")
        return Histogram(data_type=self.data_type, counts=self.counts * other, bins=self.bins,
                         road_type=self.road_type)

    def plot(self, title="", show=True, save_dir=None):
        """
        Plots the histogram
        :param title: String, title for the plot.
        :param show: Bool, run plt.show(block=False)?
        :param save_dir: String, directory where the plot should be saved.
        :return:
        """
        if self.counts is None or self.bins is None:
            raise ValueError("No data stored that can be plotted. Use histogram.calculate first.")

        plt.figure()
        plt.bar(x=self.bins.get_array()[:-1], height=self.counts, width=self.bins.step_size, align='edge', ec='black')
        if self.road_type is RoadType.UNKNOWN:
            plt.title(title)
        else:
            plt.title("[" + self.road_type.name.lower() + "]: " + title)
        plt.xlabel(self.data_type.german_name + " / " + self.data_type.unit)
        if self.data_type.id in TRACK_WISE_VARIABLES:
            plt.ylabel("Anzahl")
        else:
            plt.ylabel("Sekunden mit Messwert")
        if show:
            plt.show(block=False)
        if save_dir is not None:
            if not os.path.isdir(os.path.join("output", save_dir)):
                os.mkdir(os.path.join("output", save_dir))
            plt.savefig(os.path.join("output", save_dir, "1DHistogram_" + self.data_type.var_name + ".png"), dpi=166)

    def calculate(self, data, bins: Union[int, Bins], bins_digits=0):
        """
        Calculates the histogram for a given data. As counts the seconds with a specific value is used.
        :param data: array of Datapoints, time series of a physical quantity.
        :param bins: int (number of desired bins of histogram) or Bins object
        :param bins_digits: int, number of digits to which the start and end of bins is rounded,
                            if the bins are calculated automatically
        :return:
        """
        # Determine bins of histogram
        if type(bins) == int and bins > 0:
            data_values = [dp.value for dp in data]
            max_value = self.round_to_significant_digits(max(data_values), bins_digits)
            min_value = self.round_to_significant_digits(min(data_values), bins_digits)
            bin_array = np.linspace(min_value, max_value, num=bins + 1)
            self.bins = Bins(min(bin_array), max(bin_array), np.diff(bin_array))
            hist_bins = self.bins.get_array()
        elif type(bins) is Bins:
            self.bins = bins
            hist_bins = self.bins.get_array()
        else:
            raise ValueError("bins must be a positive integer or a Bins object!")

        # Calculate histogram
        counts = np.zeros(np.size(hist_bins) - 1)
        data.sort(key=lambda x: x.timestamp, reverse=False)

        if len(data) > 1:
            # Determine the measurement frequency
            datetime_delta_ts = [data[i + 1].timestamp - data[i].timestamp
                                 for i in range(len(data) - 1)]
            delta_ts = [datetime_delta_t.seconds + datetime_delta_t.microseconds / 1000000
                        for datetime_delta_t in datetime_delta_ts]
            measurement_frequency = self.round_to_significant_digits(float(np.median(delta_ts)), 1)

            for i in range(len(data) - 1):
                # Filter out gaps in the data
                delta_t = delta_ts[i]
                if delta_t > measurement_frequency:
                    continue

                # Determine bin
                current_value = data[i].value
                bin_num = None
                for b in range(len(hist_bins) - 1):
                    if hist_bins[b] <= current_value < hist_bins[b + 1]:
                        bin_num = b
                        break

                if bin_num is not None:
                    if self.data_type.id in TRACK_WISE_VARIABLES:
                        counts[bin_num] += 1
                    else:
                        counts[bin_num] += delta_t
                else:
                    raise HistogramError("No bin could be found for " + str(current_value) + "!")
        elif self.data_type.id in TRACK_WISE_VARIABLES:  # Some histograms are track-wise and have therefore no duration, they count per definition as one and must be evaluated correctly in the postprocessing!
            # Determine bin
            current_value = data[0].value
            bin_num = None
            for b in range(len(hist_bins) - 1):
                if hist_bins[b] <= current_value < hist_bins[b + 1]:
                    bin_num = b
                    break

            if bin_num is not None:
                counts[bin_num] += 1
            else:
                raise HistogramError("No bin could be found for " + str(current_value) + "!")

        self.counts = counts

    def to_pickle(self, subfolder=""):
        """
        Saves data to pickle file.
        :param subfolder: String, if you want to save the data into a specific folder. Leading / mandatory!
        """
        pickle_dict = {
            'data_type': self.data_type.var_name,
            'road_type': str(self.road_type).lower(),
            'bins': self.bins.get_array(),
            'counts': self.counts
        }
        if not os.path.isdir(os.path.join("output", subfolder)):
            os.mkdir(os.path.join("output", subfolder))
        with open(os.path.join("output", subfolder, "1DHistogram_" + self.data_type.var_name + ".pickle"), "wb") as f:
            pickle.dump(pickle_dict, f)

    @staticmethod
    def round_to_significant_digits(value: float, digits=2):
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


class Histogram2D:
    def __init__(self, x_data_type, y_data_type, road_type: RoadType, counts=None, x_bins: Bins = None,
                 y_bins: Bins = None):
        self.x_data_type = x_data_type
        self.y_data_type = y_data_type
        self.counts = counts
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.road_type = road_type

    def __add__(self, other):
        if not type(other) is Histogram2D:
            raise TypeError("You can only sum two Histogram2D objects.")
        if not self.x_data_type == other.x_data_type:
            raise ArithmeticError("Histograms can only be combined, if type of data in x is the same.")
        if not self.y_data_type == other.y_data_type:
            raise ArithmeticError("Histograms can only be combined, if type of data in y is the same.")
        if not self.x_bins == other.x_bins:
            raise ArithmeticError("Histograms can only be combined, if the bins in x are the same.")
        if not self.y_bins == other.y_bins:
            raise ArithmeticError("Histograms can only be combined, if the bins in y are the same.")
        if not self.road_type == other.road_type:
            raise ArithmeticError("Histograms can only be combined, if road_type of data is the same.")
        return Histogram2D(x_data_type=self.x_data_type,
                           y_data_type=self.y_data_type,
                           counts=self.counts + other.counts,
                           x_bins=self.x_bins,
                           y_bins=self.y_bins,
                           road_type=self.road_type)

    def plot_2d(self, title="", show=True, save_dir=None):
        """
        Plots the 2D histogram as 2D colormap chart.
        :param title: Title for the plot.
        :param show: Bool, run plt.show(block=False)?
        :param save_dir: String, directory where the plot should be saved.
        :return:
        """
        if self.counts is None or self.x_bins is None or self.y_bins is None:
            raise ValueError("No data stored that can be plotted. Use histogram.calculate first.")

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot matrix
        cax = ax.matshow(self.counts.T, interpolation='nearest', aspect="auto")
        colorbar = fig.colorbar(cax)
        colorbar.ax.set_ylabel("Sekunden mit Messwert")

        # Define axis ticks
        if len(self.x_bins) - 1 > 10:
            plt.xticks(np.arange(-0.5, len(self.x_bins) - 1, (len(self.x_bins) - 1) / 10),
                       np.around(np.linspace(self.x_bins.minimum, self.x_bins.maximum, 11), 2))
        else:
            plt.xticks(np.arange(-0.5, len(self.x_bins) - 1, 1), self.x_bins.get_array())

        if len(self.y_bins) - 1 > 10:
            plt.yticks(np.arange(-0.5, len(self.y_bins) - 1, (len(self.y_bins) - 1) / 10),
                       np.around(np.linspace(self.y_bins.minimum, self.y_bins.maximum, 11), 2))
        else:
            plt.yticks(np.arange(-0.5, len(self.y_bins) - 1, 1), self.y_bins.get_array())

        # Define title and axis labels
        if self.road_type is RoadType.UNKNOWN:
            plt.title(title)
        else:
            plt.title("[" + self.road_type.name.lower() + "]: " + title)
        plt.xlabel(self.x_data_type.german_name + " / " + self.x_data_type.unit)
        plt.ylabel(self.y_data_type.german_name + " / " + self.y_data_type.unit)
        if show:
            plt.show(block=False)
        if save_dir is not None:
            if not os.path.isdir(os.path.join("output", save_dir)):
                os.mkdir(os.path.join("output", save_dir))
            plt.savefig(os.path.join("output", save_dir, "2DHistogram_" + self.x_data_type.var_name + "_"
                                     + self.y_data_type.var_name + "_2Dplot.png"), dpi=166)

    def plot_3d(self, title="", show=True, save_dir=None):
        """
        Plots the 2D histogram as 3D bar chart.
        :param title: Title for the plot.
        :param show: Bool, run plt.show(block=False)?
        :param save_dir: String, directory where the plot should be saved.
        :return:
        """
        if self.counts is None or self.x_bins is None or self.y_bins is None:
            raise ValueError("No data stored that can be plotted. Use histogram.calculate first.")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Calculte positions of bars
        xpos, ypos = np.meshgrid(self.x_bins.get_array()[:-1], self.y_bins.get_array()[:-1])
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)

        # Determine length, width and height of bars
        dx = self.x_bins.step_size
        dy = self.y_bins.step_size
        dz = self.counts.T.flatten()

        # Colormap can be added, if wished
        # cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
        # max_height = np.max(dz)  # get range of colorbars so we can normalize
        # min_height = np.min(dz)
        # # scale each z to [0,1], and get their rgb values
        # rgba = [cmap((k - min_height) / max_height) for k in dz]

        # Plot 3D bars
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

        # Define title and axis labels
        if self.road_type is RoadType.UNKNOWN:
            plt.title(title)
        else:
            plt.title("[" + self.road_type.name.lower() + "]: " + title)
        plt.xlabel(self.x_data_type.german_name + " / " + self.x_data_type.unit)
        plt.ylabel(self.y_data_type.german_name + " / " + self.y_data_type.unit)
        ax.set_zlabel("Sekunden mit Messwert")
        if show:
            plt.show(block=False)
        if save_dir is not None:
            if not os.path.isdir(os.path.join("output", save_dir)):
                os.mkdir(os.path.join("output", save_dir))
            plt.savefig(os.path.join("output", save_dir, "2DHistogram_" + self.x_data_type.var_name + "_"
                                     + self.y_data_type.var_name + "_3Dplot.png"), dpi=166)

    def calculate(self, data_x, data_y, bins_x: Union[int, Bins], bins_y: Union[int, Bins], bin_digits: tuple = (0, 0)):
        """
        Calculates the histogram for a given data. As counts the seconds with a specific value is used.
        :param data_x: array of Datapoints, time series of a physical quantity for the x-axis
        :param data_y: array of Datapoints, time series of a physical quantity for the y-axis
        :param bins_x: int (number of desired bins of histogram) or Bins object describing the bins for the x-axis
        :param bins_y: int (number of desired bins of histogram) or Bins object describing the bins for the y-axis
        :param bin_digits: tuple, number of digits to which the start and end of bins is rounded,
                           if the bins are calculated automatically
        :return:
        """
        # Determine bins of x axis of the histogram
        if type(bins_x) == int and bins_x > 0:
            data_values = [dp.value for dp in data_x]
            max_value = self.round_to_significant_digits(max(data_values), bin_digits[0])
            min_value = self.round_to_significant_digits(min(data_values), bin_digits[0])
            bin_array = np.linspace(min_value, max_value, num=bins_x + 1)
            self.x_bins = Bins(min(bin_array), max(bin_array), np.diff(bin_array))
            hist_bins_x = self.x_bins.get_array()
        elif type(bins_x) is Bins:
            self.x_bins = bins_x
            hist_bins_x = self.x_bins.get_array()
        else:
            raise ValueError("bins_x must be a positive integer or a Bins object!")

        # Determine bins of y axis of the histogram
        if type(bins_y) == int and bins_y > 0:
            data_values = [dp.value for dp in data_x]
            max_value = self.round_to_significant_digits(max(data_values), bin_digits[1])
            min_value = self.round_to_significant_digits(min(data_values), bin_digits[1])
            bin_array = np.linspace(min_value, max_value, num=bins_y + 1)
            self.y_bins = Bins(min(bin_array), max(bin_array), np.diff(bin_array))
            hist_bins_y = self.y_bins.get_array()
        elif type(bins_y) is Bins:
            self.y_bins = bins_y
            hist_bins_y = self.y_bins.get_array()
        else:
            raise ValueError("bins_y must be a positive integer or a Bins object!")

        # Preprocess data (interpolate data in the smaller dataset, to get two data arrays of the same length)
        # Assumption: value is constant until next datapoint
        if len(data_x) > len(data_y):
            bigger_dataset = data_x
            smaller_dataset = data_y
            x_bigger = True
        else:
            bigger_dataset = data_y
            smaller_dataset = data_x
            x_bigger = False

        # Sort lists chronologically
        bigger_dataset.sort(key=lambda x: x.timestamp, reverse=False)
        smaller_dataset.sort(key=lambda x: x.timestamp, reverse=False)

        if len(bigger_dataset) > 10000:
            # Create batches
            batches = []
            end = 0
            while end < len(bigger_dataset):
                if end + 10000 < len(bigger_dataset):
                    batches.append(bigger_dataset[end:end + 10000])
                else:
                    batches.append(bigger_dataset[end:-1])
                end = end + 10000

            # Process batches parallelized
            pool = mp.Pool(THREADS)
            logging.info("--> Preparing data for 2D histogram ({} batches asynchronously)...".format(len(batches)))
            start_time = time.time()
            result_objects = [pool.apply_async(Histogram2D._prepare_data, args=(i, batch, smaller_dataset)) for i, batch
                              in enumerate(batches)]

            results = []
            for r in result_objects:
                results.append((r.get()[0], r.get()[1], r.get()[2]))

            pool.close()
            pool.join()

            results.sort(key=lambda x: x[0])
            new_bigger_dataset = []
            new_smaller_dataset = []
            for r in results:
                new_bigger_dataset.extend(r[1])
                new_smaller_dataset.extend(r[2])

        else:
            logging.info("--> Preparing data for 2D histogram...")
            start_time = time.time()
            _, new_bigger_dataset, new_smaller_dataset = Histogram2D._prepare_data(0, bigger_dataset, smaller_dataset)

        end_time = time.time()
        dt = end_time - start_time
        logging.info("Finished preparations. It took {} hours, {} minutes and {} seconds.".format(round(dt / 3600),
                                                                                                  round(
                                                                                                      (dt % 3600) / 60),
                                                                                                  np.around(
                                                                                                      (dt % 3600) % 60,
                                                                                                      3)))

        # Collect prepared data for histogram
        if x_bigger:
            x_data_for_hist = [dp for dp in new_bigger_dataset]
            y_data_for_hist = [dp for dp in new_smaller_dataset]
        else:
            x_data_for_hist = [dp for dp in new_smaller_dataset]
            y_data_for_hist = [dp for dp in new_bigger_dataset]

        # Clear up memory
        del bigger_dataset
        del smaller_dataset
        del new_bigger_dataset
        del new_smaller_dataset

        # Calculate histogram
        counts = np.zeros((np.size(hist_bins_x) - 1, np.size(hist_bins_y) - 1))
        x_data_for_hist.sort(key=lambda x: x.timestamp, reverse=False)
        y_data_for_hist.sort(key=lambda x: x.timestamp, reverse=False)

        # Determine the measurement frequency
        datetime_delta_ts = [x_data_for_hist[i + 1].timestamp - x_data_for_hist[i].timestamp
                             for i in range(len(x_data_for_hist) - 1)]
        delta_ts = [datetime_delta_t.seconds + datetime_delta_t.microseconds / 1000000
                    for datetime_delta_t in datetime_delta_ts]
        measurement_frequency = self.round_to_significant_digits(float(np.median(delta_ts)), 1)

        for i in range(len(x_data_for_hist) - 1):
            # Filter out gaps in the data
            delta_t = delta_ts[i]
            if delta_t > measurement_frequency:
                continue

            # Determine bin
            current_value_x = x_data_for_hist[i].value
            current_value_y = y_data_for_hist[i].value
            bin_x = bin_y = None
            for b_x in range(len(hist_bins_x) - 1):
                if hist_bins_x[b_x] <= current_value_x < hist_bins_x[b_x + 1]:
                    bin_x = b_x
                    break
            for b_y in range(len(hist_bins_y) - 1):
                if hist_bins_y[b_y] <= current_value_y < hist_bins_y[b_y + 1]:
                    bin_y = b_y
                    break
            if bin_x is not None and bin_y is not None:
                counts[bin_x][bin_y] += delta_t
            else:
                raise HistogramError("No bin could be found for (x,y) = (" + str(current_value_x) + ","
                                     + str(current_value_y) + ")!")

        # Save results
        self.counts = counts

    def to_pickle(self, subfolder=""):
        """
        Saves data to pickle file.
        :param subfolder: String, if you want to save the data into a specific folder. Leading / mandatory!
        """
        pickle_dict = {
            'data_type_x': self.x_data_type.var_name,
            'data_type_y': self.y_data_type.var_name,
            'road_type': str(self.road_type).lower(),
            'bins_x': self.x_bins,
            'bins_y': self.y_bins,
            'counts': self.counts
        }
        filename = os.path.join("output", subfolder, "2DHistogram_" + self.x_data_type.var_name + "_" + \
                                self.y_data_type.var_name + ".pickle")
        with open(filename, "wb") as f:
            pickle.dump(pickle_dict, f)

    @staticmethod
    def round_to_significant_digits(value: float, digits=2):
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
    def _datetime_to_timestamp(dt):
        return calendar.timegm(dt.timetuple())

    @staticmethod
    def _prepare_data(it, bigger_dataset_batch, smaller_dataset):
        new_bigger_dataset = []
        new_smaller_dataset = []
        for dp in bigger_dataset_batch:
            # Remove all data from the bigger dataset, where no data from the smaller dataset is available
            # (before first entry of smaller dataset)
            if dp.timestamp < smaller_dataset[0].timestamp:
                continue

            for i_smaller in range(len(smaller_dataset)):
                if smaller_dataset[i_smaller].timestamp > dp.timestamp:
                    new_bigger_dataset.append(dp)
                    smaller_dp_i1 = smaller_dataset[i_smaller]
                    smaller_dp_i = smaller_dataset[i_smaller - 1]

                    # If the value keeps constant
                    if smaller_dp_i.value == smaller_dp_i1.value:
                        new_smaller_dataset.append(Datapoint(smaller_dp_i.varname, smaller_dp_i.value,
                                                             dp.timestamp, smaller_dp_i.road_type))
                    # Else interpolate linearly between the two datapoints
                    else:
                        interpolated_value = np.interp([Histogram2D._datetime_to_timestamp(dp.timestamp)],
                                                       [Histogram2D._datetime_to_timestamp(smaller_dp_i.timestamp),
                                                        Histogram2D._datetime_to_timestamp(smaller_dp_i1.timestamp)],
                                                       [float(smaller_dp_i.value), float(smaller_dp_i1.value)])
                        new_smaller_dataset.append(Datapoint(smaller_dp_i.varname, interpolated_value,
                                                             dp.timestamp, smaller_dp_i.road_type))

                    # smaller_dp_before = smaller_dataset[i_smaller - 1]
                    # new_smaller_dataset.append(Datapoint(smaller_dp_before.varname, smaller_dp_before.value,
                    #                                      dp.timestamp, smaller_dp_before.road_type))
                    break
        return it, new_bigger_dataset, new_smaller_dataset
