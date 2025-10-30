also entweder binneselect x.*, vd.name from (
select vehicle_id,stop_time - start_time > '00:00:01' as  time_is_valid, count(*) 
from track.charging tc
where tc.vehicle_id in (1700000988,1700000989,1700000990,1700000991,1700000992,1700000635,1700000296)
group by vehicle_id, stop_time - start_time > '00:00:01' 
order by time_is_valid, vehicle_id) x
join fleet_test.vehicle_data vd on x.vehicle_id = vd.vehicle_id