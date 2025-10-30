select time, value
from sensor.can 
where request_id = 13704
and vehicle_id in (1700000635)
limit %(limit)s
