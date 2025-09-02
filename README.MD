# Electric Vehicle Usage Data Set for Battery Aging Studies

This repository contains a comprehensive dataset collected to quantify real-world stresses seen by battery electric vehicles and especially the traction battery.
It accompanies the following article:

**Lab-to-Field Gap in Battery Aging Studies: Mismatch of Operating Conditions between Laboratory Environments and Real-world Automotive Applications**

by Markus Schreiber, Lukas Köning, Georg Balke, Kareem Abo Gamra, Jonas Kayl, Brian Dietermann, Raphael Urban, Cristina Grosu and Markus Lienkamp

Technical University of Munich (TUM), Institute of Automotive Technology.

## Overview

The main aim of the article is to investigate the gap between laboratory battery aging studies and real-world battery electric vehicle (BEV) operation.
The longevity of traction batteries is critical for the widespread adoption of electric vehicles. While accelerated aging tests in laboratories are commonly used to study battery degradation, these tests often apply stress factors that differ significantly from real-world operating conditions. This dataset provides detailed real-world EV operational data alongside a comprehensive literature review of laboratory aging studies, enabling researchers to better understand and bridge the lab-to-field gap.


## Important: Git LFS Required

To download the parquet files in this repository, please ensure that Git LFS is installed on your system. For installation instructions and more information, visit [Git LFS](https://git-lfs.com/).

## Dataset Description

The dataset includes:

- **Field Data:**  
  Real-world operational data from seven unmodified series-production electric vehicles (two Volkswagen ID.3 Pro Performance and five CUPRA Born) collected over multiple years and covering more than 72,000 km of driving in public road traffic in Germany. The data includes:
  - Battery pack and cell-level parameters (SOC, voltage, temperature, charge/discharge rates)
  - Vehicle motion data (trip duration, speed, distance)
  - Charging sessions and parking periods
  - Ambient temperature during operation

- **Data Acquisition Methodology:**  
  Data was collected primarily via Unified Diagnostic Services (UDS) over the standardized On-Board Diagnostics (OBD) II interface using a custom data logger.

- **Data Processing:**  
  Raw data was processed into histograms of relevant signals for efficient aggregation and analysis. The dataset includes JSON files representing these histograms for trips, chargings, and other operational states. The JSON files can be utilized to directly recreate the figures from the associated article.


## Dataset Structure

The dataset is organized as follows:

```
├── data
│   ├── value_overview.csv              # Overview of the dataset values and associated value_ids: important for parquet data
│   ├── json                           # Directory containing JSON histogram files
│   │   ├── charging_*.json            # Charging session histograms per vehicle and trip
│   │   ├── track_*.json               # Driving session histograms per vehicle and trip
│   ├── uds_data                      # Raw UDS data in parquet format per vehicle - all measurements combined per vehicle
│   │   ├── {vehicle_id}.parquet
│   │   └── ...
├── LICENSE                          # Dataset license information
├── main.ipynb                       # Jupyter notebook for data analysis quick start
├── README.MD                        # This README file
```

- The `json` directory contains processed histogram data for individual charging and driving sessions, derived solely from UDS data. They can be used to reproduce the analyses and figures presented in the article.
- The `uds_data` directory contains raw vehicle data logs collected via UDS.

## Fleet Specifications

| Vehicle Model           | Manufacturer | Year | Number of Vehicles | Battery Pack Configuration | Net Energy Capacity |
|------------------------|--------------|------|--------------------|----------------------------|---------------------|
| ID.3 Pro Performance   | Volkswagen   | 2020 | 2                  | 108s2p (9 modules, 216 cells) | 58 kWh              |
| CUPRA Born             | CUPRA        | 2022 | 5                  | 108s2p (9 modules, 216 cells) | 58 kWh              |

The vehicles feature a synchronous motor at the rear axle with 70 kW continuous and 150 kW peak power.


## License

The dataset is distributed under the Creative Commons Attribution - NonCommercial 4.0 International (CC BY-NC 4.0) license. When using or adapting the data, please cite the paper which links back to this repository.


## Citation

If you use this dataset, please cite the associated article:

 Schreiber, Markus and Köning, Lukas and Balke, Georg and Abo Gamra, Kareem and Kayl, Jonas and Dietermann, Brian and Urban, Raphael and Grosu, Cristina and Lienkamp, Markus: "Lab-to-Field Gap in Battery Aging Studies: Mismatch of Operating Conditions between Laboratory Environments and Real-world Automotive Applications.", 2025 (under review at eTransportation) DOI: [to be added]


```
@article{schreiber2025,
  title={Lab-to-Field Gap in Battery Aging Studies: Mismatch of Operating Conditions between Laboratory Environments and Real-world Automotive Applications},
  author={Schreiber, Markus and Köning, Lukas and Balke, Georg and Abo Gamra, Kareem and Kayl, Jonas and Dietermann, Brian and Urban, Raphael and Grosu, Cristina and Lienkamp, Markus},
  year={2025},
  journal={under review at eTransportation}
}
```


## Contact

For questions or further information, please contact:

Markus Schreiber  
E-Mail: markus.schreiber@tum.de

