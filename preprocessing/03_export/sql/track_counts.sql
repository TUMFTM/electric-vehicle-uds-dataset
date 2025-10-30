select x.*, vd.name from (
select vehicle_id,stop_time - start_time > '00:00:01' as  time_is_valid, count(*) 
from track.track 
group by vehicle_id, stop_time - start_time > '00:00:01' 
order by time_is_valid, vehicle_id) x
join fleet_test.vehicle_data vd on x.vehicle_id = vd.vehicle_id