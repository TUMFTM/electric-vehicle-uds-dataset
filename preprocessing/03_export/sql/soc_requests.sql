select vehicle_id, time, value as soc from (
select vehicle_id -1700000000 as vehicle_id, time, value,
value - lag(value, 1) over (partition by vehicle_id order by time asc) as delta_soc
from sensor.can 
where request_id = 13704
and vehicle_id in (1700000635,1700000296)
)x where delta_soc != 0