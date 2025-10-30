select time, value
from sensor.can 
where request_id = 13707
and vehicle_id in (1700000635)
order by time asc
limit %(limit)s