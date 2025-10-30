select charging_id, vehicle_id - 1700000000 as vehicle_id, counts, bins_id_x
from  track.charging_ev_histograms teh
join track.charging t on teh.charging_id = t.id 
where value_id_x = %(value_id)s and value_id_y = 0
and vehicle_id in (1700000988,1700000989,1700000990,1700000991,1700000992,1700000635,1700000296)