select vehicle_id, currentsoc_pct as soc, carcapturedtimestamp as time from (
select vehicle_id - 1700000000 as vehicle_id, currentsoc_pct, carcapturedtimestamp, currentsoc_pct - lag(currentsoc_pct, 1) over (partition by car order by carcapturedtimestamp asc) as delta_soc
from comfficient_share.cupras_batterystatus cb 
join fleet_test.vehicle_data vd   on replace(replace(car,'-',''),' ','')  = replace(replace(name,'-',''),' ','')
order by car, carcapturedtimestamp asc
)x where delta_soc != 0