select cv.* from (
select distinct value_id_x
from track.track_ev_histograms
where value_id_y = 0) v
join vehicle.can_value cv on v.value_id_x = cv.value_id;