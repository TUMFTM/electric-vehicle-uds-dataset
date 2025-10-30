-- sensor.can definition

-- Drop table

-- DROP TABLE sensor.can;

CREATE TABLE sensor.can (
	vehicle_id int8 NOT NULL,
	"time" timestamp NOT NULL,
	request_id int8 NULL,
	value numeric(8, 2) NULL,
	CONSTRAINT pkey_can PRIMARY KEY (vehicle_id, "time"),
	CONSTRAINT can_fk FOREIGN KEY (request_id) REFERENCES vehicle.can_request(request_id),
	CONSTRAINT fkey_can_user_vehicle FOREIGN KEY (vehicle_id) REFERENCES fleet_test.vehicle(id) ON DELETE RESTRICT ON UPDATE CASCADE
)
PARTITION BY RANGE ("time");
CREATE INDEX fki_can_ ON ONLY sensor.can USING btree (vehicle_id);