select id as charging_id, vehicle_id - 1700000000 as vehicle_id,
stop_time - start_time > '00:00:01' as  time_is_valid, 
start_soc < stop_soc as soc_is_valid, 
start_time,stop_time,start_soc,stop_soc,avg_power,max_power,min_temp,max_temp
from track.charging tc
where tc.vehicle_id in (1700000988,1700000989,1700000990,1700000991,1700000992,1700000635,1700000296)
order by time_is_valid, vehicle_id, start_time