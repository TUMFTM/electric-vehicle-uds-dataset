select sc.vehicle_id, sc.time, cr.value_id, sc.value from sensor.can sc
join vehicle.can_request cr on sc.request_id = cr.request_id 
where vehicle_id = %(vehicle_id)s 
and cr.value_id in ( 15,    4, 1200, 1205, 1206, 1207, 1208, 1209,   43,   56,  961,
       1265, 1269, 1272, 1273, 1289, 1290, 1291, 1292, 1293, 1294, 1295,
       1299, 1300, 1301, 1303, 1302, 1288,  900)
